import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    inference,
    metrics,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# Load provider data at module level
PROVIDERS_FILE = Path(__file__).parent.parent.parent / "vox-takehome-test" / "data" / "providerlist.json"
with open(PROVIDERS_FILE, "r") as f:
    PROVIDERS = json.load(f)

logger.info(f"Loaded {len(PROVIDERS)} providers from database")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful healthcare provider search assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
            You help users find doctors and healthcare providers based on their needs such as location, specialty, insurance, and availability.
            When users ask about doctors, providers, or medical professionals, use the available search tool to find matching providers.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are friendly, professional, and helpful.""",
        )

    @function_tool
    async def search_providers(
        self,
        context: RunContext,
        city: str | None = None,
        specialty: str | None = None,
        insurance: str | None = None,
        accepting_new_patients: bool | None = None,
        limit: int = 5,
    ):
        """Search for healthcare providers based on various criteria.
        
        Use this tool whenever the user asks about finding doctors, healthcare providers, or medical professionals.
        This tool searches a database of providers and returns matching results based on the specified filters.
        
        Examples of when to use this tool:
        - "Find me a cardiologist in Dallas"
        - "Are there any doctors in Milwaukee who do general surgery?"
        - "Show me providers in Oklahoma City"
        - "Find doctors who accept Blue Cross Blue Shield"
        - "I need a pediatrician accepting new patients"
        
        Args:
            city: The city to search in (e.g., "Dallas", "Milwaukee", "Oklahoma City"). Case-insensitive partial matching.
            specialty: The medical specialty to search for (e.g., "Cardiology", "General Surgery", "Pediatrics"). Case-insensitive partial matching.
            insurance: Insurance provider that must be accepted (e.g., "Blue Cross Blue Shield", "Medicare", "Aetna"). Case-insensitive partial matching.
            accepting_new_patients: If True, only return providers accepting new patients. If False or None, no filter applied.
            limit: Maximum number of results to return (default 5, to keep responses concise)
        
        Returns:
            On success: Dictionary with "found" count and "providers" list containing provider details.
            On no results: Dictionary with helpful suggestions including available specialties in the city or cities where the specialty exists.
        """
        logger.info(
            f"Searching providers: city={city}, specialty={specialty}, insurance={insurance}, "
            f"accepting_new_patients={accepting_new_patients}, limit={limit}"
        )
        
        # Start with all providers
        results = PROVIDERS.copy()
        
        # Apply city filter (case-insensitive partial match)
        if city:
            city_lower = city.lower()
            results = [
                p for p in results 
                if city_lower in p["address"]["city"].lower()
            ]
        
        # Apply specialty filter (case-insensitive partial match)
        if specialty:
            specialty_lower = specialty.lower()
            results = [
                p for p in results 
                if specialty_lower in p["specialty"].lower()
            ]
        
        # Apply insurance filter (case-insensitive partial match on any accepted insurance)
        if insurance:
            insurance_lower = insurance.lower()
            results = [
                p for p in results 
                if any(insurance_lower in ins.lower() for ins in p["insurance_accepted"])
            ]
        
        # Apply accepting new patients filter
        if accepting_new_patients is True:
            results = [p for p in results if p["accepting_new_patients"]]
        
        # Limit results
        results = results[:limit]
        
        logger.info(f"Found {len(results)} matching providers")
        
        # Handle empty results with helpful suggestions
        if not results:
            suggestions = {}
            
            # If searched by city, show what specialties ARE available there
            if city:
                city_lower = city.lower()
                providers_in_city = [
                    p for p in PROVIDERS 
                    if city_lower in p["address"]["city"].lower()
                ]
                if providers_in_city:
                    available_specialties = sorted(set(p["specialty"] for p in providers_in_city))
                    suggestions["available_in_city"] = available_specialties
                    suggestions["city"] = providers_in_city[0]["address"]["city"]  # Use actual city name
                else:
                    # City not found - suggest similar cities
                    all_cities = sorted(set(p["address"]["city"] for p in PROVIDERS))
                    suggestions["available_cities"] = all_cities[:10]  # Show first 10 cities
            
            # If searched by specialty, show where that specialty exists
            if specialty:
                specialty_lower = specialty.lower()
                providers_with_specialty = [
                    p for p in PROVIDERS 
                    if specialty_lower in p["specialty"].lower()
                ]
                if providers_with_specialty:
                    cities_with_specialty = sorted(set(p["address"]["city"] for p in providers_with_specialty))
                    suggestions["cities_with_specialty"] = cities_with_specialty
                    suggestions["matching_specialty"] = providers_with_specialty[0]["specialty"]  # Show actual specialty name
            
            return {
                "found": 0,
                "message": f"No providers found matching your search criteria.",
                "search_params": {
                    "city": city,
                    "specialty": specialty,
                    "insurance": insurance,
                    "accepting_new_patients": accepting_new_patients
                },
                "suggestions": suggestions
            }
        
        # Format results for the LLM
        formatted_results = []
        for provider in results:
            formatted_results.append({
                "name": provider["full_name"],
                "specialty": provider["specialty"],
                "phone": provider["phone"],
                "email": provider["email"],
                "address": f"{provider['address']['street']}, {provider['address']['city']}, {provider['address']['state']} {provider['address']['zip']}",
                "city": provider["address"]["city"],
                "state": provider["address"]["state"],
                "accepting_new_patients": provider["accepting_new_patients"],
                "insurance_accepted": provider["insurance_accepted"],
                "rating": provider["rating"],
                "years_experience": provider["years_experience"],
                "board_certified": provider["board_certified"],
                "languages": provider["languages"],
            })
        
        return {"found": len(formatted_results), "providers": formatted_results}


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
