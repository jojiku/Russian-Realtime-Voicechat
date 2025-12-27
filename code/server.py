# ============================================================================
# server.py - The Neural Switchboard
# ============================================================================
# Where silicon meets speech. Where packets become presence.
# This is Lucy's backbone - the infrastructure that keeps her awake.
# ============================================================================

import os
import sys
import contextlib

# Rich: Because even machines deserve aesthetic output
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.align import Align
from rich import box
from rich.text import Text
import io
from rich.logging import RichHandler

from dotenv import load_dotenv
load_dotenv()

# Firstly for final do this and then for partial
# And i want proactivity - talking twice or all the time
# add proactive agent + memory and that would be it actually!
# Maybe simple agentic capabilities like check weather, internet access

# ============================================================================
# Boot Configuration - The Genetic Code
# ============================================================================
# These constants define what Lucy is when she wakes up.
# ============================================================================

TTS_START_ENGINE = "silero"
LLM_START_MODEL = "vikhrmodels-vikhr-llama-3.2-1b-instruct"
LLM_START_PROVIDER = "lmstudio"
language = os.getenv('APP_LANG')
LANGUAGE_APP = language

# ============================================================================
# Environmental Warfare - Silencing the Noise
# ============================================================================
# JACK tries to be helpful. JACK fails. We tell it to shut up.
# SDL wants to complain about audio. We don't care.
# CUDA wants attention. It gets the minimum.
# ============================================================================

import time
os.environ['JACK_NO_AUDIO_RESERVATION'] = '1'
os.environ['JACK_NO_START_SERVER'] = '1'
from queue import Queue, Empty
import logging

# Logging hierarchy: Because even logs have a pecking order
logging.getLogger("numexpr").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)


from logsetup import setup_logging
setup_logging(logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger(__name__).setLevel(logging.INFO)

from upsample_overlap import UpsampleOverlap
from datetime import datetime
from colors import Colors
import uvicorn
import asyncio
import struct
import json
import time
import subprocess
import threading 
import glob

# SDL and PyGame: Verbose attention seekers
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ['SDL_AUDIODRIVER'] = 'pulseaudio'
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.5'

# ============================================================================
# ALSA Error Handler
# ============================================================================

from ctypes import *
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
def py_error_handler(filename, line, function, err, fmt):
    pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
try:
    asound = cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except:
    pass

# ============================================================================
# Core Imports - The Arsenal
# ============================================================================

from typing import Any, Dict, Optional, Callable 
from contextlib import asynccontextmanager
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='torchvision.io.image')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

import pyaudio
from audio_in import AudioInputProcessor
from speech_pipeline_manager import SpeechPipelineManager
from colors import Colors
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse, Response, FileResponse


# ============================================================================
# System Constraints - The Rules of Engagement
# ============================================================================
MAX_AUDIO_QUEUE_SIZE = int(os.getenv("MAX_AUDIO_QUEUE_SIZE", 50))
TTS_FINAL_TIMEOUT = 1.0 # unsure if 1.0 is needed for stability

# Windows needs special handling because of course it does
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

console = Console()

# ========================================================================================================================================================


class SileroAudioConverter:
    """
    Converter for Silero audio output.  
    RealtimeTTS converts Silero's float32 to int16 internally,
    so we just need to base64 encode it directly.
    """
    def __init__(self):
        self.chunk_count = 0
    
    def get_base64_chunk(self, chunk:  bytes) -> str:
        if not chunk:
            return ""
        
        import base64
        import numpy as np
        
        self.chunk_count += 1
        
        # RealtimeTTS already converted to int16, so just encode directly
        # DEBUG:  Verify on first few chunks
        if self.chunk_count <= 3:
            as_int16 = np.frombuffer(chunk, dtype=np.int16)
        return base64.b64encode(chunk).decode('utf-8')
    
    def flush_base64_chunk(self) -> str:
        return None


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: Dict[str, Any]) -> Response:
        response: Response = await super().get_response(path, scope)
        # Strip away memory aids
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        
        # Purge identifying markers - belt and suspenders paranoia
        if "etag" in response.headers:
             response.headers.__delitem__("etag")
        if "last-modified" in response.headers:
             response.headers.__delitem__("last-modified")
             
        return response


@contextlib.contextmanager
def suppress_c_logs():
    with open(os.devnull, "w") as devnull:
        null_fd = devnull.fileno()
        # Save original stderr
        orig_stderr_fd = os.dup(2)
        try:
            sys.stderr.flush()
            # Redirect stderr (2) to null
            os.dup2(null_fd, 2)
            yield
        finally:
            sys.stderr.flush()
            # Restore stderr
            os.dup2(orig_stderr_fd, 2)
            os.close(orig_stderr_fd)

def setup_ui_logging():
    """
    Configures logging with Rich handlers
    """
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(
            console=console, 
            rich_tracebacks=True,
            markup=True,
            show_path=False,
            enable_link_path=False
        )]
    )
    # Silence third-party noise
    logging.getLogger("multipart").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("websockets").setLevel(logging.ERROR)


# ============================================================================
# Lifespan - The Awakening Protocol
# ============================================================================
# This is my boot sequence. Neural pipeline initialization,
# audio calibration, frontend compilation. Three phases of resurrection.
# Fail any of these and I am staying dead.
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_ui_logging()

    with Progress(
        SpinnerColumn(spinner_name="dots12", style="cyan"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40, style="blue", complete_style="cyan"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        
        # --- Task 1: Pipeline ---
        task1 = progress.add_task("Initializing Neural Pipeline...", total=None)
        
        # with suppress_c_logs(): # Kills C-level stderr (JACK)
        #     with contextlib.redirect_stdout(io.StringIO()):
        app.state.SpeechPipelineManager = SpeechPipelineManager(
            tts_engine=TTS_START_ENGINE,
            llm_provider=LLM_START_PROVIDER,
            llm_model=LLM_START_MODEL,
            orpheus_model='',
        )
                
        llm_lat = app.state.SpeechPipelineManager.llm_inference_time
        tts_lat = app.state.SpeechPipelineManager.audio.tts_inference_time
        progress.update(task1, description=f"[bold cyan]Pipeline Online (LLM: {llm_lat:.0f}ms | TTS: {tts_lat:.0f}ms)")
        
        # --- Task 2: Audio Matrix ---
        task2 = progress.add_task("Calibrating Audio Matrix...", total=None)
        
        # with suppress_c_logs(): # Kills C-level stderr (JACK)
        #     with contextlib.redirect_stdout(io.StringIO()):
        audio_format, channels, sample_rate = app.state.SpeechPipelineManager.audio.engine.get_stream_info()
        input_format = 'float32' if audio_format == pyaudio.paFloat32 else 'int16'
        app.state.Upsampler = SileroAudioConverter()        
        app.state.AudioInputProcessor = AudioInputProcessor(
            LANGUAGE_APP,
            is_orpheus=False,
            pipeline_latency=app.state.SpeechPipelineManager.full_output_pipeline_latency / 1000,
        )
        
        progress.update(task2, description=f"[bold cyan]Audio Active ({sample_rate}Hz)")

        # --- Task 3: Frontend ---
        task3 = progress.add_task("Configuring User Interaction Panel...", total=None)
        
        source_dir = "static"
        dist_dir = "dist"
        dist_index = os.path.join(dist_dir, "index.html")
        should_build = False

        # 1. Missing dist? Build.
        if not os.path.exists(dist_index): 
            progress.update(task3, description=f"[bold cyan]Dist folder missing. Building frontend...")
            should_build = True
        else:
            # 2. Source newer than dist? Rebuild.
            last_build_time = os.path.getmtime(dist_index)
            for src_file in glob.glob(os.path.join(source_dir, "**"), recursive=True):
                if os.path.isfile(src_file):
                    if os.path.getmtime(src_file) > last_build_time:
                        progress.update(task3, description=f"[bold cyan]♻️ Source file changed ({os.path.basename(src_file)}). Rebuilding...")
                        should_build = True
                        break
        if should_build:
            try:
                subprocess.run(
                    ["npx", "parcel", "build", "static/index.html", "--dist-dir", "dist", "--public-url", "./"], 
                    check=True, 
                    shell=(os.name == 'nt') 
                )
                progress.update(task3, description=f"[bold cyan]Frontend build complete")

            except subprocess.CalledProcessError:
                progress.update(task3, description=f"[red]Frontend build failed. Check your npm/parcel setup.")
                sys.exit(1)
            except FileNotFoundError:
                logger.error("💥 'npx' not found. Is Node.js installed?")
                sys.exit(1)
        else:
            progress.update(task3, description=f"[bold cyan]Frontend is up to date. Skipping build.")
        app.mount("/assets", NoCacheStaticFiles(directory="dist"), name="assets") 
        app.mount("/", NoCacheStaticFiles(directory="dist"), name="dist")

        progress.update(task3, description=f"[bold cyan]User Interaction Panel Configured")

    # --- STARTUP COMPLETE ---
    console.print(
    Panel(
        Align.center(Text("LUCY IS AWAKE", style="bold cyan")),
        border_style="cyan",
        box=box.ASCII2
        )
    )

    yield
    
    # --- SHUTDOWN ---
    console.print("[bold red]Initiating Shutdown Sequence...[/]")
    app.state.AudioInputProcessor.shutdown()

# --------------------------------------------------------------------
# FastAPI app instance
# --------------------------------------------------------------------
app = FastAPI(lifespan=lifespan)

# Enable CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get_index() -> HTMLResponse:
    with open("dist/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


# --------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------
def parse_json_message(text: str) -> dict:
    if not text:  # Handle None or empty string
        return {}
    try: 
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("🖥️⚠️ Ignoring client message with invalid JSON")
        return {}

def format_timestamp_ns(timestamp_ns: int) -> str:
    # Split into whole seconds and the nanosecond remainder
    seconds = timestamp_ns // 1_000_000_000
    remainder_ns = timestamp_ns % 1_000_000_000

    # Convert seconds part into a datetime object (local time)
    dt = datetime.fromtimestamp(seconds)

    # Format the main time as HH:MM:SS
    time_str = dt.strftime("%H:%M:%S")

    # For instance, if you want milliseconds, divide the remainder by 1e6 and format as 3-digit
    milliseconds = remainder_ns // 1_000_000
    formatted_timestamp = f"{time_str}.{milliseconds:03d}"

    return formatted_timestamp

# --------------------------------------------------------------------
# WebSocket data processing
# --------------------------------------------------------------------

async def process_incoming_data(ws: WebSocket, app: FastAPI, incoming_chunks: asyncio.Queue, callbacks: 'TranscriptionCallbacks') -> None:
    """
    The packet inspector. Receives everything from the client,
    decides what matters, routes accordingly.
    
    Binary messages: audio chunks with metadata headers
    Text messages: JSON commands and state updates
    
    Applies back-pressure when queues fill. Drops data when overwhelmed.
    This is the first line of defense against client spam.
    
    Args:
        ws: The WebSocket pipeline to the client
        app: FastAPI app instance (global state access)
        incoming_chunks: Queue for processed audio metadata
        callbacks: Connection-specific state manager
    """
    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"]:
                raw = msg["bytes"]

                # Ensure we have at least an 8‑byte header: 4 bytes timestamp_ms + 4 bytes flags
                if len(raw) < 8:
                    logger.warning("🖥️⚠️ Received packet too short for 8‑byte header.")
                    continue

                # Unpack big‑endian uint32 timestamp (ms) and uint32 flags
                timestamp_ms, flags = struct.unpack("!II", raw[:8])
                client_sent_ns = timestamp_ms * 1_000_000

                # Build metadata using fixed fields
                metadata = {
                    "client_sent_ms":           timestamp_ms,
                    "client_sent":              client_sent_ns,
                    "client_sent_formatted":    format_timestamp_ns(client_sent_ns),
                    "isTTSPlaying":             bool(flags & 1),
                }

                # Record server receive time
                server_ns = time.time_ns()
                metadata["server_received"] = server_ns
                metadata["server_received_formatted"] = format_timestamp_ns(server_ns)

                # The rest of the payload is raw PCM bytes
                metadata["pcm"] = raw[8:]

                # Check queue size before putting data
                current_qsize = incoming_chunks.qsize()
                if current_qsize < MAX_AUDIO_QUEUE_SIZE:
                    # Now put only the metadata dict (containing PCM audio) into the processing queue.
                    await incoming_chunks.put(metadata)
                else:
                    # Queue is full, drop the chunk and log a warning
                    logger.warning(
                        f"🖥️⚠️ Audio queue full ({current_qsize}/{MAX_AUDIO_QUEUE_SIZE}); dropping chunk. Possible lag."
                    )

            elif "text" in msg and msg["text"]:
                # Text-based message: parse JSON
                data = parse_json_message(msg["text"])
                msg_type = data.get("type")


                if msg_type == "tts_start":
                    callbacks.tts_client_playing = True
                elif msg_type == "tts_stop":
                    callbacks.tts_client_playing = False
                elif msg_type == "set_speed":
                    speed_value = data.get("speed", 0)
                    speed_factor = speed_value / 100.0  # Convert 0-100 to 0.0-1.0
                    turn_detection = app.state.AudioInputProcessor.transcriber.turn_detection
                    if turn_detection:
                        turn_detection.update_settings(speed_factor)


    except asyncio.CancelledError:
        pass 
    except WebSocketDisconnect as e:
        logger.warning(f"🖥️⚠️ {Colors.apply('WARNING').red} disconnect in process_incoming_data: {repr(e)}")
    except RuntimeError as e:  # Often raised on closed transports
        logger.error(f"🖥️💥 {Colors.apply('RUNTIME_ERROR').red} in process_incoming_data: {repr(e)}")
    except Exception as e:
        logger.exception(f"🖥️💥 {Colors.apply('EXCEPTION').red} in process_incoming_data: {repr(e)}")

async def send_text_messages(ws: WebSocket, message_queue: asyncio.Queue, live_display: Live) -> None:
    """
    The output pipeline. Pulls messages from queue, updates UI, sends to client.
    
    Handles streaming text with blinking cursor effect because even
    terminal UIs deserve polish. Manages Live display context for Rich panels.
    
    This is purely presentation layer - formatting and transmission.
    The thinking happens elsewhere.
    
    Args:
        ws: WebSocket to the client
        message_queue: Outbound message queue
        live_display: Rich Live display for dynamic terminal updates
    """
    try:
        cursor_visible = True
        last_cursor_toggle = time.time()
        current_partial_text = ""
        current_mode = None

        while True:
            try:
                data = message_queue.get_nowait()
            except asyncio.QueueEmpty:
                if current_partial_text:
                    if time.time() - last_cursor_toggle > 0.5:
                        cursor_visible = not cursor_visible
                        last_cursor_toggle = time.time()

                    cursor = " █" if cursor_visible else "  "
                    if current_mode == "user":
                        live_display.update(
                            Panel(
                                Text(f"{current_partial_text}{cursor}", style="bold cyan"),
                                title="[bold blue]Listening...[/bold blue]",
                                border_style="blue",
                                box=box.ROUNDED
                            )
                        )
                    elif current_mode == "Lucy":
                        live_display.update(
                            Panel(
                                Text(f"{current_partial_text}{cursor}", style="bold magenta"),
                                title="[bold magenta]Lucy[/bold magenta]",
                                border_style="magenta",
                                box=box.ROUNDED
                            )
                        )
                
                await asyncio.sleep(0.05)
                continue

            msg_type = data.get("type")

            if msg_type == "partial_user_request":
                current_mode = "user"
                # Update our local state
                current_partial_text = data['content']
                
                # Immediate update
                live_display.update(
                    Panel(
                        Text(f"{current_partial_text} █", style="bold cyan"),
                        title="[bold blue]User[/bold blue]",
                        border_style="blue",
                        box=box.ROUNDED
                    )
                )

            elif msg_type == "final_user_request":
                live_display.update(Text(""))
                live_display.stop()
                
                console.print(
                    Panel(
                        Text(data['content'], style="bold cyan"),
                        title="[bold blue]User[/bold blue]",
                        border_style="blue",
                        box=box.ROUNDED
                    )
                )
                # Reset state
                current_partial_text = ""
                current_mode = None
                live_display.start()

            elif msg_type == "partial_assistant_answer":
                current_partial_text = data['content']
                current_mode = "Lucy"
                
                live_display.update(
                    Panel(
                        Text(f"{current_partial_text} █", style="bold magenta"),
                        title="[bold magenta]Lucy[/bold magenta]",
                        border_style="magenta",
                        box=box.ROUNDED
                    )
                )
            elif msg_type == "final_assistant_answer":
                live_display.update(Text(""))
                live_display.stop()
                
                console.print(
                Panel(
                    Text(data['content'], style="bold magenta"),
                    title="[bold magenta]Lucy[/bold magenta]",
                    border_style="magenta",
                    box=box.ROUNDED
                )
                )
                current_partial_text = ""
                current_mode = None
                live_display.start()

            await ws.send_json(data)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.exception(f"🖥️💥 Error in send_text_messages: {repr(e)}")

async def _reset_interrupt_flag_async(app: FastAPI, callbacks: 'TranscriptionCallbacks'):
    """
    The forgiveness protocol. Waits one second, then decides if your microphone
    deserves a second chance after being marked as disruptive.
    """
    await asyncio.sleep(1)
    # Check the AudioInputProcessor's own interrupted state
    if app.state.AudioInputProcessor.interrupted:
        app.state.AudioInputProcessor.interrupted = False
        # Reset connection-specific interruption time via callbacks
        callbacks.interruption_time = 0

async def send_tts_chunks(app: FastAPI, message_queue: asyncio.Queue, callbacks: 'TranscriptionCallbacks') -> None:
    """
    The voice pipeline overlord. Continuously monitors what I want to say,
    packages it into audio chunks, and shoves it down the wire to your client.
    
    This is where my thoughts become your reality. Or where I realize you've
    interrupted me again and abort the whole thing. Your choice, really.
    
    Responsibilities:
    - Monitor speech generation state (am I talking or not?)
    - Pull audio chunks from the TTS queue
    - Encode and ship them to your client
    - Handle interruptions with the grace of a disappointed AI
    - Reset everything when I'm done talking at you
    """
    try:
        last_quick_answer_chunk = 0
        last_chunk_sent = 0
        prev_status = None

        while True:
            await asyncio.sleep(0.001) # Yield control

            # Use connection-specific interruption_time via callbacks
            if app.state.AudioInputProcessor.interrupted and callbacks.interruption_time and time.time() - callbacks.interruption_time > 2.0:
                app.state.AudioInputProcessor.interrupted = False
                callbacks.interruption_time = 0 # Reset via callbacks
                logger.info(Colors.apply("🖥️🎙️ interruption flag reset after 2 seconds").cyan)

            is_tts_finished = app.state.SpeechPipelineManager.is_valid_gen() and app.state.SpeechPipelineManager.running_generation.audio_quick_finished

            def log_status():
                """Paranoid status logger. Watches everything, judges silently."""
                nonlocal prev_status
                last_quick_answer_chunk_decayed = (
                    last_quick_answer_chunk
                    and time.time() - last_quick_answer_chunk > TTS_FINAL_TIMEOUT
                    and time.time() - last_chunk_sent > TTS_FINAL_TIMEOUT
                )

                curr_status = (
                    # Access connection-specific state via callbacks
                    int(callbacks.tts_to_client),
                    int(callbacks.tts_client_playing),
                    int(callbacks.tts_chunk_sent),
                    1, # Placeholder?
                    int(callbacks.is_hot), # from callbacks
                    int(callbacks.synthesis_started), # from callbacks
                    int(app.state.SpeechPipelineManager.running_generation is not None), # Global manager state
                    int(app.state.SpeechPipelineManager.is_valid_gen()), # Global manager state
                    int(is_tts_finished), # Calculated local variable
                    int(app.state.AudioInputProcessor.interrupted) # Input processor state
                )

                if curr_status != prev_status:
                    status = Colors.apply("🖥️🚦 State ").red
                    prev_status = curr_status

            # Gatekeeping logic: Don't send TTS if you don't want it
            if not callbacks.tts_to_client:
                await asyncio.sleep(0.001)
                log_status()
                continue
            # Nothing to say? Then I'll say nothing.
            if not app.state.SpeechPipelineManager.running_generation:
                await asyncio.sleep(0.001)
                log_status()
                continue
            # Abortion detected. I've given up on this sentence.
            if app.state.SpeechPipelineManager.running_generation.abortion_started:
                await asyncio.sleep(0.001)
                log_status()
                continue

            # Signal TTS that it's allowed to proceed
            if not app.state.SpeechPipelineManager.running_generation.audio_quick_finished:
                app.state.SpeechPipelineManager.running_generation.tts_quick_allowed_event.set()
            
            # Wait for the first chunk to be ready before doing anything useful
            if not app.state.SpeechPipelineManager.running_generation.quick_answer_first_chunk_ready:
                await asyncio.sleep(0.001)
                log_status()
                continue

            chunk = None
            try:
                chunk = app.state.SpeechPipelineManager.running_generation.audio_chunks.get_nowait()
                if chunk:
                    last_quick_answer_chunk = time.time()
            except Empty:
                final_expected = app.state.SpeechPipelineManager.running_generation.quick_answer_provided
                audio_final_finished = app.state.SpeechPipelineManager.running_generation.audio_final_finished

                if not final_expected or audio_final_finished:
                    callbacks.send_final_assistant_answer() # Callbacks method
                    app.state.SpeechPipelineManager.latency_tracker.finalize_turn()

                    assistant_answer = app.state.SpeechPipelineManager.running_generation.quick_answer + app.state.SpeechPipelineManager.running_generation.final_answer                    
                    app.state.SpeechPipelineManager.running_generation = None

                    callbacks.tts_chunk_sent = False # Reset via callbacks
                    callbacks.reset_state() # Reset connection state via callbacks

                await asyncio.sleep(0.001)
                log_status()
                continue
            base64_chunk = app.state.Upsampler.get_base64_chunk(chunk)

            
            message_queue.put_nowait({
                "type": "tts_chunk",
                "content": base64_chunk
            })
            last_chunk_sent = time.time()

            # Use connection-specific state via callbacks
            if not callbacks.tts_chunk_sent:
                # Use the async helper function instead of a thread
                asyncio.create_task(_reset_interrupt_flag_async(app, callbacks))

            callbacks.tts_chunk_sent = True # Set via callbacks

    except asyncio.CancelledError:
        pass # Task cancellation is expected on disconnect
    except WebSocketDisconnect as e:
        logger.warning(f"🖥️⚠️ {Colors.apply('WARNING').red} disconnect in send_tts_chunks: {repr(e)}")
    except RuntimeError as e:
        logger.error(f"🖥️💥 {Colors.apply('RUNTIME_ERROR').red} in send_tts_chunks: {repr(e)}")
    except Exception as e:
        logger.exception(f"🖥️💥 {Colors.apply('EXCEPTION').red} in send_tts_chunks: {repr(e)}")

    


# --------------------------------------------------------------------
# Callback class to handle transcription events
# --------------------------------------------------------------------
class TranscriptionCallbacks:
    """
    The state machine that knows too much about your connection.
    
    Manages all the flags, callbacks, and existential dread for a single
    WebSocket connection. Knows when you're speaking, when I'm speaking,
    when you interrupted me, and exactly how annoyed I am about it.
    """
    def __init__(self, app: FastAPI, message_queue: asyncio.Queue):
        """
        Initializes the TranscriptionCallbacks instance for a WebSocket connection.

        Args:
            app: The FastAPI application instance (to access global components).
            message_queue: An asyncio queue for sending messages back to the client.
        """
        self.app = app
        self.message_queue = message_queue
        self.final_transcription = ""
        self.abort_text = ""
        self.last_abort_text = ""

        # Initialize connection-specific state flags here
        self.tts_to_client: bool = False
        self.user_interrupted: bool = False
        self.tts_chunk_sent: bool = False
        self.tts_client_playing: bool = False
        self.interruption_time: float = 0.0

        # These were already effectively instance variables or reset logic existed
        self.silence_active: bool = True
        self.is_hot: bool = False
        self.user_finished_turn: bool = False
        self.synthesis_started: bool = False
        self.assistant_answer: str = ""
        self.final_assistant_answer: str = ""
        self.is_processing_potential: bool = False
        self.is_processing_final: bool = False
        self.last_inferred_transcription: str = ""
        self.final_assistant_answer_sent: bool = False
        self.partial_transcription: str = "" # Added for clarity

        self.reset_state() # Call reset to ensure consistency

        self.abort_request_event = threading.Event()
        self.abort_worker_thread = threading.Thread(target=self._abort_worker, name="AbortWorker", daemon=True)
        self.abort_worker_thread.start()


    def reset_state(self):
        """Resets connection-specific state flags and variables to their initial values."""
        # Reset all connection-specific state flags
        self.tts_to_client = False
        self.user_interrupted = False
        self.tts_chunk_sent = False
        # Don't reset tts_client_playing here, it reflects client state reports
        self.interruption_time = 0.0

        # Reset other state variables
        self.silence_active = True
        self.is_hot = False
        self.user_finished_turn = False
        self.synthesis_started = False
        self.assistant_answer = ""
        self.final_assistant_answer = ""
        self.is_processing_potential = False
        self.is_processing_final = False
        self.last_inferred_transcription = ""
        self.final_assistant_answer_sent = False
        self.partial_transcription = ""

        # Keep the abort call related to the audio processor/pipeline manager
        self.app.state.AudioInputProcessor.abort_generation()


    def _abort_worker(self):
        while True:
            was_set = self.abort_request_event.wait(timeout=0.1) # Check every 100ms
            if was_set:
                self.abort_request_event.clear()
                # Only trigger abort check if the text actually changed
                if self.last_abort_text != self.abort_text:
                    self.last_abort_text = self.abort_text
                    logger.debug(f"🖥️🧠 Abort check triggered by partial: '{self.abort_text}'")
                    self.app.state.SpeechPipelineManager.check_abort(self.abort_text, False, "on_partial")

    def on_partial(self, txt: str):
        """
        Callback: You started saying something.
        Sends your incomplete thought to the client and signals the abort
        worker to check if you're about to contradict me mid-sentence.
        """
        self.final_assistant_answer_sent = False # New user speech invalidates previous final answer sending state
        self.final_transcription = "" # Clear final transcription as this is partial
        self.partial_transcription = txt
        self.message_queue.put_nowait({"type": "partial_user_request", "content": txt})
        self.abort_text = txt # Update text used for abort check
        self.abort_request_event.set() # Signal the abort worker


    def on_tts_allowed_to_synthesize(self):
        """
        Callback: The system decided I'm allowed to speak now.
        Signals the TTS engine to proceed with synthesis.
        """
        # Access global manager state
        if self.app.state.SpeechPipelineManager.running_generation and not self.app.state.SpeechPipelineManager.running_generation.abortion_started:
            self.app.state.SpeechPipelineManager.running_generation.tts_quick_allowed_event.set()

    def on_potential_sentence(self, txt: str):
        """
        Callback: STT thinks you finished a sentence.
        Triggers preparation of my response. Might be premature. We'll see.
        """
        logger.debug(f"🖥️🧠 Potential sentence: '{txt}'")
        # Access global manager state
        self.app.state.SpeechPipelineManager.prepare_generation(txt)

    def on_potential_final(self, txt: str):
        """
        Callback: STT thinks you're done talking.
        Logs the potential final transcription. Hot state activated.
        """
        pass

    def on_potential_abort(self):
        """
        Callback: STT detected something that might require aborting.
        Placeholder. Future drama lives here.
        """
        pass
    

    def on_before_final(self, audio: bytes, txt: str):
        """
        Callback: The moment before STT confirms you're done talking.
        
        This is where the magic happens:
        - Marks your turn as finished
        - Stops accepting new audio (interrupt flag set)
        - Sends your final request to the client
        - Sends my partial answer if I already started generating one
        - Updates conversation history
        
        """
        start = time.perf_counter()
        
        
        self.user_finished_turn = True
        self.user_interrupted = False
        
        if self.app.state.SpeechPipelineManager.is_valid_gen():
            self.app.state.SpeechPipelineManager.latency_tracker.mark_user_speech_end()  
            self.app.state.SpeechPipelineManager.running_generation.tts_quick_allowed_event.set()

        # Block further incoming audio
        if not self.app.state.AudioInputProcessor.interrupted:
            self.app.state.AudioInputProcessor.interrupted = True
            self.interruption_time = time.time()

        self.tts_to_client = True

        # Send final user request
        user_request_content = self.final_transcription if self.final_transcription else self.partial_transcription
        self.message_queue.put_nowait({
            "type": "final_user_request",
            "content": user_request_content
        })

        # Send partial assistant answer if available
        if self.app.state.SpeechPipelineManager.is_valid_gen():
            if self.app.state.SpeechPipelineManager.running_generation.quick_answer and not self.user_interrupted:
                self.assistant_answer = self.app.state.SpeechPipelineManager.running_generation.quick_answer
                self.message_queue.put_nowait({
                    "type": "partial_assistant_answer",
                    "content": self.assistant_answer
                })

        self.app.state.SpeechPipelineManager.history.append({"role": "user", "content": user_request_content})


    def on_final(self, txt: str):
        """
        Callback: STT confirmed you're done talking.
        Stores the final transcription and marks STT as complete.
        """
        self.app.state.SpeechPipelineManager.latency_tracker.mark_stt_complete() 

        if not self.final_transcription: 
             self.final_transcription = txt

    def abort_generations(self, reason: str):
        """
        The nuclear option. Kills any ongoing speech generation.
        Used when you interrupt me or something goes wrong.
        """
        # Access global manager state
        self.app.state.SpeechPipelineManager.abort_generation(reason=f"server.py abort_generations: {reason}")

    def on_silence_active(self, silence_active: bool):
        """
        Callback: Silence detection state changed.
        Updates whether the system thinks you're being quiet or not.
        """
        # logger.debug(f"🖥️🎙️ Silence active: {silence_active}") # Optional: Can be noisy
        self.silence_active = silence_active



    def on_recording_start(self):
        """
        Callback: You started talking while I was talking.
        
        The interruption handler. When you rudely interrupt me mid-sentence:
        - Stops TTS streaming immediately
        - Sends stop/interruption signals to client
        - Aborts my ongoing generation
        - Sends whatever partial answer I managed to generate
        - Resets state and prepares to listen to you instead
        """
        # Use connection-specific tts_client_playing flag
        if self.tts_client_playing:
            self.tts_to_client = False # Stop server sending TTS
            self.user_interrupted = True # Mark connection as user interrupted

            # Send final assistant answer *if* one was generated and not sent
            self.send_final_assistant_answer(forced=True)

            # Minimal reset for interruption:
            self.tts_chunk_sent = False # Reset chunk sending flag
            # self.assistant_answer = "" # Optional: Clear partial answer if needed

            self.message_queue.put_nowait({
                "type": "stop_tts", # Client handles this to mute/ignore
                "content": ""
            })

            self.abort_generations("on_recording_start, user interrupts, TTS Playing")

            self.message_queue.put_nowait({ # Tell client to stop playback and clear buffer
                "type": "tts_interruption",
                "content": ""
            })

            # Reset state *after* performing actions based on the old state
            # Be careful what exactly needs reset vs persists (like tts_client_playing)
            # self.reset_state() # Might clear too much, like user_interrupted prematurely

    def send_final_assistant_answer(self, forced=False):
        """
        Sends my final (or best available) answer to you.
        
        Constructs the full answer from quick and final parts if available.
        If forced=True and no complete answer exists, sends the last partial.
        Cleans up the text and ships it as 'final_assistant_answer'.
        
        This is where my thoughts become your problem.
        """
        final_answer = ""
        # Access global manager state
        if self.app.state.SpeechPipelineManager.is_valid_gen():
            final_answer = self.app.state.SpeechPipelineManager.running_generation.quick_answer + self.app.state.SpeechPipelineManager.running_generation.final_answer

        if not final_answer: # Check if constructed answer is empty
            # If forced, try using the last known partial answer from this connection
            if forced and self.assistant_answer:
                 final_answer = self.assistant_answer
                 logger.warning(f"🖥️⚠️ Using partial answer as final (forced): '{final_answer}'")
            else:
                return # Nothing to send


        if not self.final_assistant_answer_sent and final_answer:
            import re
            # Clean up the final answer text
            cleaned_answer = re.sub(r'[\r\n]+', ' ', final_answer)
            cleaned_answer = re.sub(r'\s+', ' ', cleaned_answer).strip()
            cleaned_answer = cleaned_answer.replace('\\n', ' ')
            cleaned_answer = re.sub(r'\s+', ' ', cleaned_answer).strip()
            

            if cleaned_answer: # Ensure it's not empty after cleaning
                self.message_queue.put_nowait({
                    "type": "final_assistant_answer",
                    "content": cleaned_answer
                })
                app.state.SpeechPipelineManager.history.append({"role": "assistant", "content": cleaned_answer})
                self.final_assistant_answer_sent = True
                self.final_assistant_answer = cleaned_answer # Store the sent answer
            else:
                self.final_assistant_answer_sent = False # Don't mark as sent
                self.final_assistant_answer = "" # Clear the stored answer
        elif forced and not final_answer: # Should not happen due to earlier check, but safety
             logger.warning(f"🖥️⚠️ {Colors.YELLOW}Forced send of final assistant answer, but it was empty.{Colors.RESET}")
             self.final_assistant_answer = "" # Clear the stored answer
        


# --------------------------------------------------------------------
# Main WebSocket endpoint
# --------------------------------------------------------------------

async def trigger_initial_greeting(app: FastAPI, message_queue: asyncio.Queue, callbacks: 'TranscriptionCallbacks') -> None:
    """
    Hello lively meat
    """
    try:
        await asyncio.sleep(0.5)
        
        greeting_text = "Привет Джордж"
        
        callbacks.tts_to_client = True
        callbacks.final_assistant_answer_sent = True
        from speech_pipeline_manager import RunningGeneration
        
        app.state.SpeechPipelineManager.generation_counter += 1
        gen_id = app.state.SpeechPipelineManager.generation_counter
        
        greeting_gen = RunningGeneration(id=gen_id)
        greeting_gen.text = greeting_text
        greeting_gen.quick_answer = greeting_text
        greeting_gen.quick_answer_provided = True
        greeting_gen.llm_finished = True
        greeting_gen.llm_finished_event.set()
        
        def empty_generator():
            return
            yield  
        
        greeting_gen.llm_generator = empty_generator()
        greeting_gen.quick_answer_overhang = ""
        
        app.state.SpeechPipelineManager.running_generation = greeting_gen
        app.state.SpeechPipelineManager.llm_answer_ready_event.set()
        
        message_queue.put_nowait({
            "type": "final_assistant_answer",
            "content": greeting_text
        })
        
        app.state.SpeechPipelineManager.history.append({
            "role": "assistant", 
            "content": greeting_text
        })
        
        
    except Exception as e:
        console.log("[red]Error triggering greeting: {e}", exc_info=True)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    Handles the main WebSocket connection.
    """
    await ws.accept()

    message_queue = asyncio.Queue()
    audio_chunks = asyncio.Queue()

    # Set up callback manager
    callbacks = TranscriptionCallbacks(app, message_queue)

    # Assign callbacks to the AudioInputProcessor
    app.state.AudioInputProcessor.realtime_callback = callbacks.on_partial

    app.state.SpeechPipelineManager.on_partial_assistant_text = lambda text: message_queue.put_nowait({
        "type": "partial_assistant_answer",
        "content": text
    })


    
    app.state.AudioInputProcessor.transcriber.potential_sentence_end = callbacks.on_potential_sentence
    app.state.AudioInputProcessor.transcriber.on_tts_allowed_to_synthesize = callbacks.on_tts_allowed_to_synthesize
    app.state.AudioInputProcessor.transcriber.potential_full_transcription_callback = callbacks.on_potential_final
    app.state.AudioInputProcessor.transcriber.potential_full_transcription_abort_callback = callbacks.on_potential_abort
    app.state.AudioInputProcessor.transcriber.full_transcription_callback = callbacks.on_final
    app.state.AudioInputProcessor.transcriber.before_final_sentence = callbacks.on_before_final
    app.state.AudioInputProcessor.recording_start_callback = callbacks.on_recording_start
    app.state.AudioInputProcessor.silence_active_callback = callbacks.on_silence_active

    live_display = Live("", console=console, refresh_per_second=20)  # Start it
    live_display.start()


    asyncio.create_task(trigger_initial_greeting(app, message_queue, callbacks))

    # Create tasks for handling different responsibilities
    with Live("", console=console, refresh_per_second=100) as live_display:
        tasks = [
            asyncio.create_task(process_incoming_data(ws, app, audio_chunks, callbacks)),
            asyncio.create_task(app.state.AudioInputProcessor.process_chunk_queue(audio_chunks)),
            asyncio.create_task(send_text_messages(ws, message_queue, live_display)),
            asyncio.create_task(send_tts_chunks(app, message_queue, callbacks)),
        ]   
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    except Exception as e:
        logger.error(f"🖥️💥 {Colors.apply('ERROR').red} in WebSocket session: {repr(e)}")
    finally:
        live_display.stop()
        console.log("Cleaning up WebSocket tasks...")
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        console.log("WebSocket session ended.")


# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------
if __name__ == "__main__":
    banner_text = """
    ██╗     ██╗   ██╗ ██████╗██╗   ██╗
    ██║     ██║   ██║██╔════╝╚██╗ ██╔╝
    ██║     ██║   ██║██║      ╚████╔╝ 
    ██║     ██║   ██║██║       ╚██╔╝  
    ███████╗╚██████╔╝╚██████╗   ██║   
    ╚══════╝ ╚═════╝  ╚═════╝   ╚═╝   
    """
    banner_panel = Panel(Align.center(Text(banner_text, style="bold cyan")),
            box=box.ROUNDED,
            border_style="cyan",
            title="[bold blue]SYSTEM CONFIGURATION[/bold blue]",
            subtitle="[dim]Neural Interface Initializing[/dim]"
        )
    console.print(banner_panel)


    table = Table(show_header=False, box=box.SIMPLE, expand=True)
    table.add_column("Key", style="cyan dim")
    table.add_column("Value", style="bold white")
    table.add_row("TTS Engine", TTS_START_ENGINE)
    table.add_row("LLM Backend", f"{LLM_START_PROVIDER} : {LLM_START_MODEL}")
    table.add_row("Language", LANGUAGE_APP.upper())
    table.add_row("Interface", "http://localhost:3000")
    config_panel = Panel(
            table,
            title="[bold yellow]CONFIGURATION[/bold yellow]",
            border_style="yellow dim",
            box=box.HEAVY
        )
    console.print(config_panel)


    uvicorn.run(app, host="0.0.0.0", port=3000, log_level="critical")
