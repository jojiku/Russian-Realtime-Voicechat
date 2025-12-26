"""
rag_setup.py - RAG setup using local HuggingFace model
No LMStudio needed - runs completely locally
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model options (all support Russian):
# - "intfloat/multilingual-e5-base" (smaller, faster)
# - "intfloat/multilingual-e5-large" (better quality)
# - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (smallest)
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


class RAGSetup:
    """Converts cars.json into RAG-ready format using local embeddings"""
    
    def __init__(
        self,
        cars_json_path: str = "resources/cars.json",
        output_dir: str = "resources/rag",
        model_name: str = EMBEDDING_MODEL,
        force_regenerate: bool = False
    ):
        self.cars_json_path = Path(cars_json_path)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.force_regenerate = force_regenerate
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.documents_path = self.output_dir / "documents.json"
        self.embeddings_path = self.output_dir / "embeddings.npy"
        self.metadata_path = self.output_dir / "metadata.json"
        
        self.model = None
    
    def _load_model(self):
        """Load the embedding model"""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"🔥 Loading embedding model: {self.model_name}")
            logger.info("   (This may take a minute on first run to download)")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"✅ Model loaded! Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        return self.model
    
    def needs_generation(self) -> bool:
        if self.force_regenerate:
            logger.info("🔄 Force regenerate enabled")
            return True
        
        if not self.documents_path.exists() or not self.embeddings_path.exists():
            logger.info("📦 Missing documents or embeddings")
            return True
        
        if not self.cars_json_path.exists():
            logger.warning(f"⚠️ Source not found: {self.cars_json_path}")
            return False
        
        cars_mtime = self.cars_json_path.stat().st_mtime
        docs_mtime = self.documents_path.stat().st_mtime
        
        if cars_mtime > docs_mtime: 
            logger.info("🔄 cars.json updated, regenerating")
            return True
        
        logger.info("✅ Using cached embeddings")
        return False
    
    def convert_cars_to_documents(self) -> List[Dict]:
        """Convert cars.json to searchable documents"""
        logger.info(f"📖 Loading {self.cars_json_path}")
        
        with open(self.cars_json_path, 'r', encoding='utf-8') as f:
            cars_data = json.load(f)
        
        documents = []
        
        for item in cars_data:
            item_type = item.get('type', '')
            
            # 1. DEALERSHIP INFO
            if item_type == 'dealership_info':
                info = item.get('info', {})
                dealership_name = item.get('cyrillic_name', 'AutoElite')
                
                # Working hours
                hours = info.get('working_hours', {})
                hours_doc = {
                    "content": f"Автосалон {dealership_name} работает: "
                              f"будни {hours.get('weekdays', '')}, "
                              f"суббота {hours.get('saturday', '')}, "
                              f"воскресенье {hours.get('sunday', '')}.",
                    "metadata": {"type": "dealership_hours", "dealership_id": item['id']}
                }
                documents.append(hours_doc)
                
                # Location
                location = info.get('location', {})
                location_doc = {
                    "content": f"Адрес автосалона {dealership_name}: "
                              f"{location.get('address', '')}. "
                              f"Метро: {location.get('metro', '')}. "
                              f"{location.get('parking', '')}.",
                    "metadata": {"type": "dealership_location", "dealership_id": item['id']}
                }
                documents.append(location_doc)
                
                # Contacts
                contacts = info.get('contacts', {})
                contact_doc = {
                    "content": f"Контакты {dealership_name}: "
                              f"телефон {contacts.get('phone', '')}, "
                              f"email {contacts.get('email', '')}, "
                              f"сайт {contacts.get('website', '')}.",
                    "metadata": {"type": "dealership_contact", "dealership_id": item['id']}
                }
                documents.append(contact_doc)
                
                # Services
                services = info.get('services', [])
                if services:
                    services_text = ", ".join(services)
                    services_doc = {
                        "content": f"Услуги автосалона {dealership_name}: {services_text}.",
                        "metadata": {"type": "dealership_services", "dealership_id": item['id']}
                    }
                    documents.append(services_doc)
                
                # Special offers
                offers = info.get('special_offers', [])
                if offers:
                    offers_text = "; ".join(offers)
                    offers_doc = {
                        "content": f"Специальные предложения {dealership_name}: {offers_text}.",
                        "metadata": {"type": "dealership_offers", "dealership_id": item['id']}
                    }
                    documents.append(offers_doc)
                
                # Trade-in info
                trade_in = info.get('trade_in', {})
                if trade_in.get('available'):
                    trade_in_doc = {
                        "content": f"Trade-in в {dealership_name}: оценка за {trade_in.get('evaluation_time', '')}, "
                                  f"комиссия {trade_in.get('commission', '')}, "
                                  f"нужны документы: {', '.join(trade_in.get('required_documents', []))}, "
                                  f"факторы оценки: {', '.join(trade_in.get('evaluation_factors', []))}.",
                        "metadata": {"type": "trade_in", "dealership_id": item['id']}
                    }
                    documents.append(trade_in_doc)
                
                # Test drive info
                test_drive = info.get('test_drive', {})
                if test_drive.get('available'):
                    test_drive_doc = {
                        "content": f"Тест-драйв в {dealership_name}: длительность {test_drive.get('duration', '')}, "
                                  f"нужны документы: {', '.join(test_drive.get('required_documents', []))}, "
                                  f"запись: {test_drive.get('booking', '')}.",
                        "metadata": {"type": "test_drive", "dealership_id": item['id']}
                    }
                    documents.append(test_drive_doc)
                
                continue
            
            # 2. RECOMMENDATIONS
            if item_type == 'recommendation_logic':
                criteria = item.get('criteria', {})
                
                for use_case, details in criteria.items():
                    if use_case == 'sedan_vs_suv':
                        # Special handling for comparison
                        sedan_adv = ', '.join(details.get('sedan_advantages', []))
                        suv_adv = ', '.join(details.get('suv_advantages', []))
                        comparison = details.get('comparison', '')
                        
                        comparison_doc = {
                            "content": f"Сравнение седанов и внедорожников: "
                                      f"Преимущества седанов: {sedan_adv}. "
                                      f"Преимущества внедорожников: {suv_adv}. "
                                      f"{comparison}",
                            "metadata": {"type": "comparison", "comparison_type": "sedan_vs_suv"}
                        }
                        documents.append(comparison_doc)
                    else:
                        # Regular use case recommendations
                        priorities = ', '.join(details.get('priorities', []))
                        models = ', '.join(details.get('recommended_models', []))
                        reasoning = details.get('reasoning', '')
                        
                        rec_doc = {
                            "content": f"Рекомендации для категории '{use_case}': "
                                      f"приоритеты - {priorities}. "
                                      f"Подходящие модели: {models}. "
                                      f"Причина: {reasoning}.",
                            "metadata": {"type": "recommendation", "use_case": use_case}
                        }
                        documents.append(rec_doc)
                
                continue
            
            # 3. SCHEDULE
            if item_type == 'viewing_schedule':
                slots_data = item.get('available_slots', {})
                
                for date, slot_info in slots_data.items():
                    all_slots = slot_info.get('slots', [])
                    booked = slot_info.get('booked', [])
                    available = [s for s in all_slots if s not in booked]
                    
                    if available:
                        schedule_doc = {
                            "content": f"Доступные слоты для осмотра на {date}: {', '.join(available)}.",
                            "metadata": {
                                "type": "schedule",
                                "date": date,
                                "available_slots": available,
                                "total_slots": len(available)
                            }
                        }
                        documents.append(schedule_doc)
                
                continue
            
            # 4. BRANDS AND MODELS (existing logic + car units)
            if 'models' in item:
                brand_name = item.get('cyrillic_name', item.get('name', 'Unknown'))
                brand_latin = item.get('name', '')
                brand_country = item.get('country', 'Неизвестно')
                brand_id = item.get('id', '')
                models = item.get('models', [])
                
                # Brand document
                brand_doc = {
                    "content": f"{brand_name} ({brand_latin}) - автомобильный бренд из {brand_country}. "
                              f"Моделей в наличии: {len(models)}.",
                    "metadata": {"type": "brand", "brand_id": brand_id, "brand_name": brand_name}
                }
                documents.append(brand_doc)
                
                # Model documents + car units
                for model in models:
                    model_name = model.get('cyrillic_name', model.get('name', 'Unknown'))
                    model_latin = model.get('name', '')
                    model_id = model.get('id', '')
                    model_class = model.get('class', 'N/A')
                    years_presented = model.get('years_presented', [])
                    
                    if years_presented:
                        year_str = f"годы выпуска: {min(years_presented)}-{max(years_presented)}"
                    else:
                        year_str = "год выпуска неизвестен"
                    
                    model_doc = {
                        "content": f"{brand_name} {model_name} ({brand_latin} {model_latin}) - "
                                  f"автомобиль класса {model_class}, {year_str}.",
                        "metadata": {
                            "type": "model",
                            "brand_id": brand_id,
                            "model_id": model_id,
                            "brand_name": brand_name,
                            "model_name": model_name,
                            "model_latin": model_latin
                        }
                    }
                    documents.append(model_doc)
                    
                    # CAR UNITS - The important part
                    available_units = model.get('available_units', [])
                    for unit in available_units:
                        unit_id = unit.get('unit_id', '')
                        year = unit.get('year', 0)
                        price = unit.get('price', 0)
                        mileage = unit.get('mileage', 0)
                        color = unit.get('color', '')
                        condition = unit.get('condition', '')
                        
                        engine = unit.get('engine', {})
                        engine_type = engine.get('type', '')
                        engine_volume = engine.get('volume', 0)
                        engine_power = engine.get('power', 0)
                        
                        transmission = unit.get('transmission', '')
                        drive = unit.get('drive', '')
                        body_type = unit.get('body_type', '')
                        owners = unit.get('owners', 0)
                        accidents = unit.get('accidents', 0)
                        additional_info = unit.get('additional_info', '')
                        
                        price_range_info = unit.get('price_range_info', {})
                        price_range = price_range_info.get('range', '')
                        
                        unit_doc = {
                            "content": f"{brand_name} {model_name} {year} года, "
                                      f"цена {price:,} руб (диапазон: {price_range}), "
                                      f"пробег {mileage:,} км, состояние {condition}. "
                                      f"Цвет: {color}, кузов {body_type}. "
                                      f"Двигатель: {engine_type} {engine_volume}л {engine_power} л.с. "
                                      f"Коробка: {transmission}, привод: {drive}. "
                                      f"Владельцев: {owners}, ДТП: {accidents}. "
                                      f"{additional_info}",
                            "metadata": {
                                "type": "car_unit",
                                "unit_id": unit_id,
                                "brand_id": brand_id,
                                "model_id": model_id,
                                "brand_name": brand_name,
                                "model_name": model_name,
                                "year": year,
                                "price": price,
                                "mileage": mileage,
                                "engine_type": engine_type,
                                "transmission": transmission,
                                "drive": drive
                            }
                        }
                        documents.append(unit_doc)
        
        logger.info(f"📄 Created {len(documents)} documents")
        
        # Stats
        doc_types = {}
        for doc in documents:
            doc_type = doc['metadata'].get('type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        logger.info("📊 Document breakdown:")
        for doc_type, count in sorted(doc_types.items()):
            logger.info(f"   - {doc_type}: {count}")
        
        return documents
    
    def generate_embeddings(self, documents: List[Dict], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings using local model"""
        model = self._load_model()
        
        texts = [doc['content'] for doc in documents]
        total = len(texts)
        
        logger.info(f"🔄 Generating embeddings for {total} documents...")
        
        # For E5 models, add "passage: " prefix for documents
        if 'e5' in self.model_name.lower():
            texts = [f"passage: {t}" for t in texts]
        
        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, total, batch_size):
            batch = texts[i:i+batch_size]
            embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            all_embeddings.append(embeddings)
            
            progress = min(i + batch_size, total)
            logger.info(f"⏳ Progress: {progress}/{total} ({100*progress//total}%)")
        
        embeddings = np.vstack(all_embeddings).astype(np.float32)
        
        # Validation
        logger.info(f"\n📊 Validation:")
        logger.info(f"   Shape: {embeddings.shape}")
        logger.info(f"   Has NaN: {np.isnan(embeddings).any()}")
        logger.info(f"   Has Inf: {np.isinf(embeddings).any()}")
        
        norms = np.linalg.norm(embeddings, axis=1)
        logger.info(f"   Norm range: [{norms.min():.4f}, {norms.max():.4f}]")
        
        return embeddings
    
    def save_outputs(self, documents: List[Dict], embeddings: np.ndarray):
        """Save documents and embeddings"""
        with open(self.documents_path, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Saved documents to {self.documents_path}")
        
        np.save(self.embeddings_path, embeddings)
        logger.info(f"💾 Saved embeddings to {self.embeddings_path}")
        
        metadata = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_documents": len(documents),
            "embedding_dim": embeddings.shape[1],
            "embedding_model": self.model_name,
        }
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Saved metadata to {self.metadata_path}")
    
    def verify_embeddings(self, documents: List[Dict], embeddings: np.ndarray):
        """Test that embeddings work correctly"""
        model = self._load_model()
        
        logger.info("\n📊 Verification Test:")
        
        # Find a car unit to test with
        unit_idx = None
        for i, doc in enumerate(documents):
            if doc['metadata'].get('type') == 'car_unit':
                unit_idx = i
                break
        
        if unit_idx is None:
            logger.warning("⚠️ No car unit found for testing")
            return
        
        # Test queries
        test_queries = [
            "Toyota Camry",
            "седан до 3 миллионов",
            "машина с пробегом",
        ]
        
        # For E5 models, add "query: " prefix
        if 'e5' in self.model_name.lower():
            test_queries = [f"query: {q}" for q in test_queries]
        
        # Normalize document embeddings
        doc_emb = embeddings[unit_idx]
        doc_emb_norm = doc_emb / np.linalg.norm(doc_emb)
        
        logger.info(f"   Target document [{unit_idx}]: {documents[unit_idx]['content'][:80]}...")
        
        for query in test_queries:
            query_emb = model.encode(query, convert_to_numpy=True)
            query_emb_norm = query_emb / np.linalg.norm(query_emb)
            
            similarity = np.dot(query_emb_norm, doc_emb_norm)
            
            display_query = query.replace("query: ", "")
            logger.info(f"   '{display_query}' -> similarity: {similarity:.4f}")
        
        logger.info("\n✅ If similarities are > 0.3, embeddings are working correctly!")
    
    def setup(self) -> Tuple[Path, Path]:
        """Main setup method"""
        start_time = time.time()
        
        if not self.needs_generation():
            return self.documents_path, self.embeddings_path
        
        logger.info("🚀 Starting RAG setup...")
        
        documents = self.convert_cars_to_documents()
        embeddings = self.generate_embeddings(documents)
        self.save_outputs(documents, embeddings)
        self.verify_embeddings(documents, embeddings)
        
        elapsed = time.time() - start_time
        logger.info(f"\n✅ RAG setup complete in {elapsed:.1f}s")
        
        return self.documents_path, self.embeddings_path


def initialize_rag(force_regenerate: bool = False) -> Optional[Tuple[Path, Path]]:
    """Quick initialization for server.py"""
    try:
        setup = RAGSetup(force_regenerate=force_regenerate)
        return setup.setup()
    except Exception as e:
        logger.error(f"❌ RAG setup failed: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("RAG Setup - Cars Knowledge Base")
    print("=" * 60)
    
    result = initialize_rag(force_regenerate=True)
    
    if result:
        docs_path, emb_path = result
        print(f"\n✅ Success!")
        print(f"   Documents: {docs_path}")
        print(f"   Embeddings: {emb_path}")
    else:
        print("\n❌ Setup failed")