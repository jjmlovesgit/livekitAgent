#tested 8/19/24 see requirements.txt
import os
import asyncio
from typing import Annotated
from dotenv import load_dotenv
import selectors

import asyncio

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm

from livekit import agents, rtc
from livekit.agents import JobContext, JobRequest, WorkerOptions, cli, tokenize, tts 
from livekit.agents.llm import (
    ChatContext,
    ChatImage,
    ChatMessage,
    ChatRole,
)
from livekit.agents.voice_assistant import AssistantContext, VoiceAssistant
from livekit.plugins import deepgram, openai, silero, azure


class MyPolicy(asyncio.DefaultEventLoopPolicy):
   def new_event_loop(self):
      selector = selectors.SelectSelector()
      return asyncio.SelectorEventLoop(selector)

asyncio.set_event_loop_policy(MyPolicy())

# Load environment variables from .env file
load_dotenv()
def reload_env_variables():
    livekit_url = os.environ.get('LIVEKIT_URL')
    livekit_api_key = os.environ.get('LIVEKIT_API_KEY')
    livekit_api_secret = os.environ.get('LIVEKIT_API_SECRET')
    eleven_api_key = os.environ.get('ELEVEN_API_KEY')
    deepgram_api_key = os.environ.get('DEEPGRAM_API_KEY')
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    speech_region = os.environ.get('AZURE_SPEECH_REGION')
    speech_key = os.environ.get('AZURE_SPEECH_KEY')
    
    return {
        'livekit_url': livekit_url,
        'livekit_api_key': livekit_api_key,
        'livekit_api_secret': livekit_api_secret,
        'eleven_api_key': eleven_api_key,
        'deepgram_api_key': deepgram_api_key,
        'openai_api_key': openai_api_key,
        'speech_region': speech_region,
        'speech_key': speech_key
    }
def print_env_variables(env_vars):
    for key, value in env_vars.items():
        if value:
            print(f"{key}: {value[:2]}...{value[-2:]}")
        else:
            print(f"{key}: None")
env_vars = reload_env_variables()
print_env_variables(env_vars)


class AssistantFunction(agents.llm.FunctionContext):
    """This class is used to define functions that will be called by the assistant."""
    @agents.llm.ai_callable(
        desc=(
            "Use this function whenever asked to evaluate an image, video, or the webcam feed being shared with you"
                 )
    )
    async def image(
        self,
        user_msg: Annotated[
            str,
            agents.llm.TypeInfo(desc="The user message that triggered this function"),
        ],
    ):
        print(f"Message triggering vision capabilities: {user_msg}")
        context = AssistantContext.get_current()
        context.store_metadata("user_msg", user_msg)

async def get_video_track(room: rtc.Room):
    """Get the first video track from the room. We'll use this track to process images."""
    video_track = asyncio.Future[rtc.RemoteVideoTrack]()
    for _, participant in room.participants.items():
        for _, track_publication in participant.tracks.items():
            if track_publication.track is not None and isinstance(
                track_publication.track, rtc.RemoteVideoTrack
            ):
                video_track.set_result(track_publication.track)
                print(f"Using video track {track_publication.track.sid}")
                break
    return await video_track

async def entrypoint(ctx: JobContext):
    print(f"Room name: {ctx.room.name}")
    chat_context = ChatContext(
        messages=[
            ChatMessage(
                role=ChatRole.SYSTEM,
                text=(
                    "Your name is Andrew. You are an assitant who is slightly sarcastic and witty. You have both voice and vision capabilities."
                    "Respond with clear and concise answers with minimal jargon.  Do not use emojis."
                ),
            )
        ]
    )
    gpt = openai.LLM(model="gpt-4o")
    latest_image: rtc.VideoFrame | None = None
    assistant = VoiceAssistant(
        vad=silero.VAD(),  # We'll use Silero's Voice Activity Detector (VAD)
        stt=deepgram.STT(),
        #stt=azure.STT(),  # We'll use Deepgram's Speech To Text (STT)
        llm=gpt, # We'll use GTP 4.0
        tts=azure.TTS(voice="en-US-AvaMultilingualNeural"),
        #tts=elevenlabs.TTS(), # Text-to-Speech #tts=openai_tts, 
        #tts=openai_tts,  # We'll use OpenAI's Text To Speech (TTS)
        fnc_ctx=AssistantFunction(),
        chat_ctx=chat_context,
    )
    chat = rtc.ChatManager(ctx.room)
    async def _answer(text: str, use_image: bool = False):
        """
        Answer the user's message with the given text and optionally the latest
        image captured from the video track.
        """
        args = {}
        if use_image and latest_image:
            args["images"] = [ChatImage(image=latest_image)]
        chat_context.messages.append(ChatMessage(role=ChatRole.USER, text=text, **args))
        stream = await gpt.chat(chat_context)
        await assistant.say(stream, allow_interruptions=True)
        await assistant.say(stream)
    @chat.on("message_received")
    def on_message_received(msg: rtc.ChatMessage):
        """This event triggers whenever we get a new message from the user."""
        if msg.message:
            asyncio.create_task(_answer(msg.message, use_image=False))
    @assistant.on("function_calls_finished")
    def on_function_calls_finished(ctx: AssistantContext):
        """This event triggers when an assistant's function call completes."""
        user_msg = ctx.get_metadata("user_msg")
        if user_msg:
            asyncio.create_task(_answer(user_msg, use_image=True))
    assistant.start(ctx.room)
    await asyncio.sleep(3)
    await assistant.say("Hey, how can I help you today?", allow_interruptions=True)
    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        video_track = await get_video_track(ctx.room)
        async for event in rtc.VideoStream(video_track):
            # We'll continually grab the latest image from the video track
            # and store it in a variable.
            latest_image = event.frame

async def request_fnc(req: JobRequest) -> None:
    await req.accept(entrypoint)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc))

