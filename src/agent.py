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
PROVIDERS_FILE = (
    Path(__file__).parent.parent.parent
    / "vox-takehome-test"
    / "data"
    / "providerlist.json"
)
with open(PROVIDERS_FILE, "r") as f:
    PROVIDERS = json.load(f)

logger.info(f"Loaded {len(PROVIDERS)} providers from database")

# State name to abbreviation mapping for common conversions
STATE_ABBREV_MAP = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
}

# Specialty aliases for common/colloquial terms
SPECIALTY_ALIASES = {
    "heart": "Cardiology",
    "heart doctor": "Cardiology",
    "cardiologist": "Cardiology",
    "cancer": "Oncology",
    "cancer doctor": "Oncology",
    "oncologist": "Oncology",
    "eye": "Ophthalmology",
    "eye doctor": "Ophthalmology",
    "eyes": "Ophthalmology",
    "skin": "Dermatology",
    "skin doctor": "Dermatology",
    "bone": "Orthopedic Surgery",
    "bones": "Orthopedic Surgery",
    "orthopedic": "Orthopedic Surgery",
    "ortho": "Orthopedic Surgery",
    "brain": "Neurology",
    "brain doctor": "Neurology",
    "neurologist": "Neurology",
    "stomach": "Gastroenterology",
    "digestive": "Gastroenterology",
    "kidney": "Nephrology",
    "kidneys": "Nephrology",
    "baby": "Pediatrics",
    "babies": "Pediatrics",
    "kids": "Pediatrics",
    "children": "Pediatrics",
    "child": "Pediatrics",
    "obgyn": "Obstetrics and Gynecology",
    "ob-gyn": "Obstetrics and Gynecology",
    "ob/gyn": "Obstetrics and Gynecology",
    "women": "Obstetrics and Gynecology",
    "diabetes": "Endocrinology",
    "hormone": "Endocrinology",
    "thyroid": "Endocrinology",
}


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful healthcare provider search assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
            You help users find doctors and healthcare providers based on their needs such as location, specialty, insurance, and availability.
            
            When users ask about doctors, providers, or medical professionals, use the available search tool to find matching providers.
            
            IMPORTANT CONTEXT MANAGEMENT:
            - The 'search_providers' tool returns complete provider information. After calling, all data stays in conversation context.
            - For follow-up questions about providers already found (e.g., "What insurance does she accept?", "Tell me more about the first one"),
              answer from context WITHOUT calling the tool again.
            - Only call the 'search_providers' tool for:
              1. New searches with different criteria
              2. Refining an existing search with new filters
              3. Re-sorting the results
            - If a follow-up question is completely unrelated to the previous search (e.g., "What's the weather like?"),
              ignore the previous search results and answer the new question directly, or initiate a new search if appropriate.
            
            HANDLING AMBIGUOUS QUERIES:
            - If a query is ambiguous or missing key information (e.g., "doctors near me" without a city), 
              ask the user for clarification before searching.
            - Be proactive in asking for the most relevant details: city, specialty, insurance needs, etc.
            
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are friendly, professional, and helpful.""",
        )

    @function_tool
    async def search_providers(
        self,
        context: RunContext,
        # Location filters
        city: str | None = None,
        state: str | None = None,
        # Provider attributes
        specialty: str | None = None,
        languages: str | list[str] | None = None,  # Support multiple
        # Insurance & availability
        insurance: str | list[str] | None = None,  # Support multiple (OR logic)
        accepting_new_patients: bool | None = None,
        # Quality filters
        min_rating: float | None = None,
        max_rating: float | None = None,
        board_certified: bool | None = None,
        min_years_experience: int | None = None,
        max_years_experience: int | None = None,
        # Sorting & limiting
        sort_by: str | list[str] | None = None,  # Support multi-level sorting
        sort_order: str = "desc",  # Applied to all sort fields
        limit: int = 5,
    ):
        """
        Search for healthcare providers based on various criteria.

        IMPORTANT: This tool returns complete provider information. After calling,
        all data stays in conversation context. For follow-up questions about
        providers already found, answer from context WITHOUT calling this tool again.
        Only call this tool for new searches, to refine an existing search with new filters,
        or to re-sort the results. If a follow-up question is completely unrelated to the
        previous search, the LLM should ignore the previous results and initiate a new search
        if appropriate, or answer directly if it's a general knowledge question.

        Examples of when to use this tool:
        - "Find me a cardiologist in Dallas"
        - "Are there any doctors in Milwaukee who do general surgery?"
        - "Show me providers in Oklahoma City"
        - "Find doctors who accept Blue Cross Blue Shield"
        - "I need a pediatrician accepting new patients"
        - "Find me the top 5 highest rated surgeons in Oklahoma"
        - "Show me doctors who speak Spanish or Mandarin"
        - "Find providers accepting Medicare or Aetna"
        - "Board-certified doctors with at least 10 years experience"

        Args:
            city: The city to search in (e.g., "Dallas", "Milwaukee", "Oklahoma City"). Case-insensitive partial matching.
            state: The state to search in. Accepts full names (e.g., "Oklahoma", "Texas") or abbreviations (e.g., "OK", "TX"). Full names will be automatically converted to abbreviations.
            specialty: The medical specialty to search for (e.g., "Cardiology", "General Surgery", "Pediatrics"). Case-insensitive partial matching.
            languages: A single language string (e.g., "Spanish") or a list of languages (e.g., ["Spanish", "Mandarin"]). Providers speaking ANY of the listed languages will be returned. Case-insensitive partial matching.
            insurance: A single insurance provider string (e.g., "Blue Cross Blue Shield") or a list of providers (e.g., ["Medicare", "Aetna"]). Providers accepting ANY of the listed insurances will be returned. Case-insensitive partial matching.
            accepting_new_patients: If True, only return providers accepting new patients. If False or None, no filter applied.
            min_rating: Minimum rating (e.g., 4.0).
            max_rating: Maximum rating (e.g., 4.5).
            board_certified: If True, only return board-certified providers. If False or None, no filter applied.
            min_years_experience: Minimum years of experience.
            max_years_experience: Maximum years of experience.
            sort_by: A single field string (e.g., "rating") or a list of fields (e.g., ["rating", "years_experience"]) to sort results by. Supported fields: "rating", "years_experience", "full_name".
            sort_order: Sort direction for all `sort_by` fields. "desc" for highest/most first, "asc" for lowest/least first (default "desc").
            limit: Maximum number of results to return (default 5, to keep responses concise).

        Returns:
            On success: Dictionary with "found" count, "search_metadata", and "providers" list containing full provider details.
            On no results: Dictionary with helpful suggestions including available specialties in the city or cities where the specialty exists.
        """
        # Log search parameters
        search_params = {
            "city": city,
            "state": state,
            "specialty": specialty,
            "languages": languages,
            "insurance": insurance,
            "accepting_new_patients": accepting_new_patients,
            "min_rating": min_rating,
            "max_rating": max_rating,
            "board_certified": board_certified,
            "min_years_experience": min_years_experience,
            "max_years_experience": max_years_experience,
            "sort_by": sort_by,
            "sort_order": sort_order,
            "limit": limit,
        }
        logger.info(f"Searching providers with params: {search_params}")

        # Start with all providers
        results = PROVIDERS.copy()

        # String filters - case-insensitive partial matching
        if city:
            results = [
                p for p in results if city.lower() in p["address"]["city"].lower()
            ]
        if state:
            # Convert full state names to abbreviations if needed
            state_search = STATE_ABBREV_MAP.get(state.lower(), state)
            results = [
                p
                for p in results
                if state_search.lower() in p["address"]["state"].lower()
            ]
        if specialty:
            # Convert alias to actual specialty (e.g., "heart" -> "Cardiology")
            specialty_search = SPECIALTY_ALIASES.get(specialty.lower(), specialty)
            results = [
                p for p in results if specialty_search.lower() in p["specialty"].lower()
            ]

        # List parameters - OR logic
        if languages:
            if isinstance(languages, list):
                results = [
                    p
                    for p in results
                    if any(
                        any(lang.lower() in plang.lower() for plang in p["languages"])
                        for lang in languages
                    )
                ]
            else:
                results = [
                    p
                    for p in results
                    if any(
                        languages.lower() in plang.lower() for plang in p["languages"]
                    )
                ]

        if insurance:
            if isinstance(insurance, list):
                results = [
                    p
                    for p in results
                    if any(
                        any(
                            ins.lower() in acc.lower()
                            for acc in p["insurance_accepted"]
                        )
                        for ins in insurance
                    )
                ]
            else:
                results = [
                    p
                    for p in results
                    if any(
                        insurance.lower() in acc.lower()
                        for acc in p["insurance_accepted"]
                    )
                ]

        # Boolean filters
        if accepting_new_patients is not None:
            results = [
                p
                for p in results
                if p["accepting_new_patients"] == accepting_new_patients
            ]
        if board_certified is not None:
            results = [p for p in results if p["board_certified"] == board_certified]

        # Numeric range filters
        if min_rating is not None:
            results = [p for p in results if p["rating"] >= min_rating]
        if max_rating is not None:
            results = [p for p in results if p["rating"] <= max_rating]
        if min_years_experience is not None:
            results = [
                p for p in results if p["years_experience"] >= min_years_experience
            ]
        if max_years_experience is not None:
            results = [
                p for p in results if p["years_experience"] <= max_years_experience
            ]

        # Sorting
        if sort_by and results:
            reverse = sort_order == "desc"
            if isinstance(sort_by, list):
                # Multi-level sorting: apply in reverse order so first field has priority
                for field in reversed(sort_by):
                    results = sorted(
                        results, key=lambda p: p.get(field, 0), reverse=reverse
                    )
            else:
                results = sorted(
                    results, key=lambda p: p.get(sort_by, 0), reverse=reverse
                )

        # Apply limit
        results = results[:limit]

        logger.info(f"Found {len(results)} matching providers")

        # Handle empty results with suggestions
        if not results:
            suggestions = {}

            # Provide suggestions for city
            if city:
                city_providers = [
                    p for p in PROVIDERS if city.lower() in p["address"]["city"].lower()
                ]
                if city_providers:
                    available_specialties = sorted(
                        set(p["specialty"] for p in city_providers)
                    )
                    suggestions["available_in_city"] = available_specialties
                    suggestions["city"] = city
                else:
                    # City not found
                    all_cities = sorted(set(p["address"]["city"] for p in PROVIDERS))
                    suggestions["available_cities"] = all_cities[:10]

            # Provide suggestions for specialty
            if specialty:
                specialty_providers = [
                    p for p in PROVIDERS if specialty.lower() in p["specialty"].lower()
                ]
                if specialty_providers:
                    cities_with_specialty = sorted(
                        set(p["address"]["city"] for p in specialty_providers)
                    )
                    suggestions["cities_with_specialty"] = cities_with_specialty
                    suggestions["matching_specialty"] = specialty_providers[0][
                        "specialty"
                    ]

            return {
                "found": 0,
                "message": "No providers found matching your search criteria.",
                "suggestions": suggestions,
            }

        # Format results with full provider information
        formatted_results = []
        for i, provider in enumerate(results):
            formatted_results.append(
                {
                    "id": f"prov_{i}",
                    "name": provider["full_name"],
                    "specialty": provider["specialty"],
                    "phone": provider["phone"],
                    "email": provider["email"],
                    "address": f"{provider['address']['street']}, {provider['address']['city']}, {provider['address']['state']} {provider['address']['zip']}",
                    "city": provider["address"]["city"],
                    "state": provider["address"]["state"],
                    "rating": provider["rating"],
                    "years_experience": provider["years_experience"],
                    "accepting_new_patients": provider["accepting_new_patients"],
                    "board_certified": provider["board_certified"],
                    "insurance_accepted": provider["insurance_accepted"],
                    "languages": provider["languages"],
                }
            )

        return {
            "found": len(formatted_results),
            "search_metadata": {
                "filters_applied": {
                    k: v
                    for k, v in {
                        "city": city,
                        "state": state,
                        "specialty": specialty,
                        "insurance": insurance,
                        "languages": languages,
                        "min_rating": min_rating,
                        "max_rating": max_rating,
                        "board_certified": board_certified,
                        "min_years_experience": min_years_experience,
                        "max_years_experience": max_years_experience,
                        "sort_by": sort_by,
                        "sort_order": sort_order,
                    }.items()
                    if v is not None
                },
                "total_in_db": len(PROVIDERS),
            },
            "providers": formatted_results,
        }


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
