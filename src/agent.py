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
from mongomock import MongoClient

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# Load provider data at module level
PROVIDERS_FILE = Path(__file__).parent.parent.parent / "vox-takehome-test" / "data" / "providerlist.json"
with open(PROVIDERS_FILE, "r") as f:
    PROVIDERS = json.load(f)

logger.info(f"Loaded {len(PROVIDERS)} providers from database")

# Initialize MongoDB collection for querying
mongo_client = MongoClient()
db = mongo_client.providers_db
providers_collection = db.providers
providers_collection.insert_many(PROVIDERS)
logger.info(f"Initialized MongoDB collection with {providers_collection.count_documents({})} providers")


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
        query: str,
        limit: int = 5,
    ):
        """Search for healthcare providers using MongoDB query syntax.

        Use this tool to search for doctors and healthcare providers. Construct MongoDB queries
        to filter by any field in the provider database.

        IMPORTANT NORMALIZATION RULES:
        - Abbreviations: Convert "SF" to "San Francisco", "NYC" to "New York City", "LA" to "Los Angeles"
        - For partial text matching: Use regex with case-insensitive option: {"specialty": {"$regex": "Surgery", "$options": "i"}}
        - The system handles nested fields with dot notation: "address.city", "address.state"

        Common MongoDB Operators:
        - Equality: {"field": "value"}
        - Regex/Contains: {"field": {"$regex": "pattern", "$options": "i"}}
        - Comparison: {"field": {"$gt": value, "$gte": value, "$lt": value, "$lte": value}}
        - In array: {"field": {"$in": ["value1", "value2"]}}
        - Element in array field: {"insurance_accepted": {"$elemMatch": {"$regex": "Medicare", "$options": "i"}}}
        - Logical: {"$and": [...], "$or": [...], "$not": {...}}

        Available fields:
        - full_name, specialty, phone, email
        - address.city, address.state, address.street, address.zip
        - rating (number), years_experience (number)
        - accepting_new_patients (boolean), board_certified (boolean)
        - insurance_accepted (array of strings), languages (array of strings)

        Query Examples:
        1. City search (normalize abbreviations!):
           {"address.city": "San Francisco"}

        2. Specialty with partial match:
           {"specialty": {"$regex": "Surgery", "$options": "i"}}

        3. Rating filter:
           {"rating": {"$gte": 4.5}}

        4. Multiple conditions (AND):
           {"$and": [{"address.city": "Dallas"}, {"specialty": {"$regex": "Cardiology", "$options": "i"}}, {"rating": {"$gte": 4.0}}]}

        5. Multiple conditions (OR):
           {"$or": [{"specialty": "Cardiology"}, {"specialty": "General Surgery"}]}

        6. Insurance search (array field):
           {"insurance_accepted": {"$elemMatch": {"$regex": "Medicare", "$options": "i"}}}

        7. Language search (array field):
           {"languages": "Spanish"}

        8. Complex query:
           {"$and": [{"address.state": "CA"}, {"board_certified": true}, {"accepting_new_patients": true}, {"rating": {"$gte": 4.0}}]}

        Args:
            query: MongoDB query as a JSON string (e.g., '{"address.city": "Dallas"}')
            limit: Maximum number of results to return (default 5)

        Returns:
            On success: Dictionary with "found" count and "providers" list.
            On no results: Dictionary with suggestions for alternative searches.
        """
        logger.info(f"Searching providers with query: {query}, limit: {limit}")

        try:
            # Parse the JSON query string
            try:
                query_dict = json.loads(query)
                logger.info(f"Parsed query: {query_dict}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON query: {query}, error: {e}")
                return {
                    "found": 0,
                    "error": f"Invalid JSON query format: {str(e)}",
                    "message": "Please provide a valid JSON query string."
                }

            # Execute MongoDB query
            cursor = providers_collection.find(query_dict).limit(limit)
            results = list(cursor)

            logger.info(f"Found {len(results)} matching providers")

            # Handle empty results with suggestions
            if not results:
                suggestions = {}

                # Extract city from query if present
                city = query_dict.get("address.city")
                if not city and "$and" in query_dict:
                    for condition in query_dict["$and"]:
                        if "address.city" in condition:
                            city = condition["address.city"]
                            break

                # Extract specialty from query if present
                specialty = None
                if "specialty" in query_dict:
                    if isinstance(query_dict["specialty"], dict) and "$regex" in query_dict["specialty"]:
                        specialty = query_dict["specialty"]["$regex"]
                    else:
                        specialty = query_dict["specialty"]
                elif "$and" in query_dict:
                    for condition in query_dict["$and"]:
                        if "specialty" in condition:
                            if isinstance(condition["specialty"], dict) and "$regex" in condition["specialty"]:
                                specialty = condition["specialty"]["$regex"]
                            else:
                                specialty = condition["specialty"]
                            break

                # Provide suggestions for city
                if city:
                    city_providers = list(providers_collection.find({"address.city": city}))
                    if city_providers:
                        available_specialties = sorted(set(p["specialty"] for p in city_providers))
                        suggestions["available_in_city"] = available_specialties
                        suggestions["city"] = city
                    else:
                        # City not found
                        all_cities = sorted(set(p["address"]["city"] for p in PROVIDERS))
                        suggestions["available_cities"] = all_cities[:10]

                # Provide suggestions for specialty
                if specialty:
                    specialty_query = {"specialty": {"$regex": specialty, "$options": "i"}}
                    specialty_providers = list(providers_collection.find(specialty_query))
                    if specialty_providers:
                        cities_with_specialty = sorted(set(p["address"]["city"] for p in specialty_providers))
                        suggestions["cities_with_specialty"] = cities_with_specialty
                        suggestions["matching_specialty"] = specialty_providers[0]["specialty"]

                return {
                    "found": 0,
                    "message": "No providers found matching your search criteria.",
                    "query": query_dict,
                    "suggestions": suggestions
                }

            # Format results
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

        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return {
                "found": 0,
                "error": f"Invalid query format: {str(e)}",
                "message": "Please use valid MongoDB query syntax."
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
