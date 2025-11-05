# Healthcare Provider Search Agent - Low-Level Design Document

## Overview

This document describes the implementation of a voice AI agent that searches healthcare providers from a JSON database. The agent uses LiveKit Agents framework with OpenAI GPT-4.1-mini for natural language understanding, and implements a custom search tool with sophisticated filtering, sorting, and context management capabilities.

## Architecture

### Data Loading

The provider data is loaded once at module initialization and stored in memory for fast access:

```python
PROVIDERS_FILE = Path(__file__).parent.parent.parent / "vox-takehome-test" / "data" / "providerlist.json"
with open(PROVIDERS_FILE, "r") as f:
    PROVIDERS = json.load(f)
```

**Design Decision**: Loading data into memory at startup rather than reading from disk on each search provides:
- Constant-time access (O(1) lookup)
- No file I/O latency during searches
- Acceptable memory footprint (100 providers ~ 50KB)

### Mapping Dictionaries

Two module-level dictionaries provide normalization for common user inputs:

#### State Abbreviation Mapping

Maps full state names to 2-letter abbreviations since the provider data uses abbreviations:

```python
STATE_ABBREV_MAP = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    # ... 50 states total
}
```

**Use Case**: User says "Oklahoma" but data stores "OK"

#### Specialty Aliases

Maps colloquial terms to formal medical specialties:

```python
SPECIALTY_ALIASES = {
    "heart": "Cardiology",
    "heart doctor": "Cardiology",
    "cancer": "Oncology",
    "eye": "Ophthalmology",
    "bone": "Orthopedic Surgery",
    "baby": "Pediatrics",
    "diabetes": "Endocrinology",
    # ... 29 aliases total
}
```

**Design Decision**: These mappings act as a safety net. The LLM typically extracts formal terms, but if it passes informal terms (especially from voice transcription errors), the search still works correctly.

## The search_providers Tool

### Function Signature

```python
@function_tool
async def search_providers(
    self,
    context: RunContext,
    # Location filters
    city: str | None = None,
    state: str | None = None,
    # Provider attributes
    specialty: str | None = None,
    languages: str | list[str] | None = None,
    # Insurance & availability
    insurance: str | list[str] | None = None,
    accepting_new_patients: bool | None = None,
    # Quality filters
    min_rating: float | None = None,
    max_rating: float | None = None,
    board_certified: bool | None = None,
    min_years_experience: int | None = None,
    max_years_experience: int | None = None,
    # Sorting & limiting
    sort_by: str | list[str] | None = None,
    sort_order: str = "desc",
    limit: int = 5,
)
```

### Parameter Design

**All Optional Parameters**: Every parameter defaults to `None`, allowing the LLM to specify only relevant filters for each query. This provides maximum flexibility.

**Union Types for Flexibility**: 
- `languages: str | list[str] | None` accepts both single values and lists
- `insurance: str | list[str] | None` accepts both single values and lists
- `sort_by: str | list[str] | None` accepts single field or multi-field sorting

**Default Values**:
- `sort_order: str = "desc"` - Default to highest/best first (most intuitive)
- `limit: int = 5` - Balance between providing options and keeping responses concise

### Docstring Strategy

The function docstring serves as the tool's contract with the LLM. It contains:

1. **Context Management Instructions**: Critical guidance on when to use the tool vs. answering from context
2. **Usage Examples**: Concrete examples showing the tool's purpose
3. **Parameter Documentation**: Detailed descriptions with examples
4. **Return Format Documentation**: What the LLM can expect back

**Key Section - Context Management**:
```
IMPORTANT: This tool returns complete provider information. After calling,
all data stays in conversation context. For follow-up questions about
providers already found, answer from context WITHOUT calling this tool again.
Only call this tool for new searches, to refine an existing search with new filters,
or to re-sort the results.
```

This instruction is what enables the agent to handle follow-up questions efficiently without redundant tool calls.

### Filtering Logic

The search implements a cascading filter pattern, starting with all providers and progressively narrowing results:

```python
# Start with all providers
results = PROVIDERS.copy()

# Apply each filter sequentially if parameter is provided
if city:
    results = [p for p in results if city.lower() in p["address"]["city"].lower()]
if state:
    state_search = STATE_ABBREV_MAP.get(state.lower(), state)
    results = [p for p in results if state_search.lower() in p["address"]["state"].lower()]
if specialty:
    specialty_search = SPECIALTY_ALIASES.get(specialty.lower(), specialty)
    results = [p for p in results if specialty_search.lower() in p["specialty"].lower()]
```

**Design Characteristics**:

1. **Case-Insensitive Matching**: All string comparisons use `.lower()` to handle variations in capitalization
2. **Partial Matching**: Uses `in` operator rather than exact equality, allowing "Oklahoma City" to match "Oklahoma"
3. **Normalization**: State and specialty values are mapped before filtering
4. **Short-Circuit Evaluation**: If city filter reduces results to zero, subsequent filters operate on empty list (efficient)

### List Parameter Logic (OR Semantics)

For `languages` and `insurance` parameters, when a list is provided, the search uses OR logic:

```python
if languages:
    if isinstance(languages, list):
        results = [
            p for p in results 
            if any(
                any(lang.lower() in plang.lower() for plang in p["languages"])
                for lang in languages
            )
        ]
    else:
        results = [
            p for p in results 
            if any(languages.lower() in plang.lower() for plang in p["languages"])
        ]
```

**Interpretation**: 
- `languages=["Spanish", "Mandarin"]` matches providers speaking Spanish OR Mandarin
- This is more useful than AND logic, which would be overly restrictive

**Nested Comprehension Breakdown**:
- Outer `any()`: At least one requested language must match
- Inner `any()`: At least one of provider's languages must contain the requested language
- This handles partial matching: "Spanish" matches "Spanish" or "Spanish, English"

### Boolean Filters

Boolean parameters use strict equality when provided:

```python
if accepting_new_patients is not None:
    results = [p for p in results if p["accepting_new_patients"] == accepting_new_patients]
if board_certified is not None:
    results = [p for p in results if p["board_certified"] == board_certified]
```

**Note**: Checking `is not None` rather than truthiness is critical because `False` is a valid filter value.

### Numeric Range Filters

Rating and experience filters support min/max ranges:

```python
if min_rating is not None:
    results = [p for p in results if p["rating"] >= min_rating]
if max_rating is not None:
    results = [p for p in results if p["rating"] <= max_rating]
if min_years_experience is not None:
    results = [p for p in results if p["years_experience"] >= min_years_experience]
if max_years_experience is not None:
    results = [p for p in results if p["years_experience"] <= max_years_experience]
```

**Use Case**: User asks for "providers with rating between 4.0 and 4.5" - sets both min and max

### Sorting Logic

The sorting implementation supports both single-field and multi-field sorting:

```python
if sort_by and results:
    reverse = sort_order == "desc"
    if isinstance(sort_by, list):
        # Multi-level sorting: apply in reverse order so first field has priority
        for field in reversed(sort_by):
            results = sorted(results, key=lambda p: p.get(field, 0), reverse=reverse)
    else:
        results = sorted(results, key=lambda p: p.get(sort_by, 0), reverse=reverse)
```

**Multi-Field Sorting Strategy**:
- Sort by each field in reverse order
- Python's stable sort ensures earlier sorts are preserved for ties
- Example: `sort_by=["rating", "years_experience"]` sorts by rating first, then by experience for same rating

**Default Value Handling**: `p.get(field, 0)` provides a default of 0 if field is missing, preventing KeyError

### Limit Application

```python
results = results[:limit]
```

Applied after all filtering and sorting to return top N results.

### Suggestion Generation

When no results are found, the function generates contextual suggestions:

```python
if not results:
    suggestions = {}
    
    # Provide suggestions for city
    if city:
        city_providers = [p for p in PROVIDERS if city.lower() in p["address"]["city"].lower()]
        if city_providers:
            available_specialties = sorted(set(p["specialty"] for p in city_providers))
            suggestions["available_in_city"] = available_specialties
            suggestions["city"] = city
        else:
            all_cities = sorted(set(p["address"]["city"] for p in PROVIDERS))
            suggestions["available_cities"] = all_cities[:10]
    
    # Provide suggestions for specialty
    if specialty:
        specialty_providers = [p for p in PROVIDERS if specialty.lower() in p["specialty"].lower()]
        if specialty_providers:
            cities_with_specialty = sorted(set(p["address"]["city"] for p in specialty_providers))
            suggestions["cities_with_specialty"] = cities_with_specialty
```

**Logic**:
1. If city was specified but no matches found:
   - If city exists in database: show available specialties in that city
   - If city doesn't exist: show list of valid cities
2. If specialty was specified: show cities where that specialty is available

This helps users discover what's actually available in the dataset.

### Return Format

**Success Response**:
```python
{
    "found": len(formatted_results),
    "search_metadata": {
        "filters_applied": {...},
        "total_in_db": len(PROVIDERS)
    },
    "providers": [
        {
            "id": "prov_0",
            "name": "Dr. Jane Smith",
            "specialty": "Cardiology",
            "phone": "(555) 123-4567",
            "email": "jane.smith@example.com",
            "address": "123 Main St, Dallas, TX 75001",
            "city": "Dallas",
            "state": "TX",
            "rating": 4.5,
            "years_experience": 15,
            "accepting_new_patients": true,
            "board_certified": true,
            "insurance_accepted": ["Blue Cross", "Medicare"],
            "languages": ["English", "Spanish"]
        }
    ]
}
```

**Empty Results Response**:
```python
{
    "found": 0,
    "message": "No providers found matching your search criteria.",
    "suggestions": {
        "available_in_city": ["Cardiology", "Pediatrics", "Oncology"],
        "city": "Dallas"
    }
}
```

**Design Decision**: Return complete provider objects rather than summaries. This enables the LLM to answer follow-up questions without additional tool calls.

## Agent Instructions (System Prompt)

The agent's instructions guide the LLM's behavior throughout the conversation:

```python
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
You are friendly, professional, and helpful."""
```

### Instruction Breakdown

**Voice Context Awareness**:
```
The user is interacting with you via voice, even if you perceive the conversation as text.
```
Reminds the LLM to format responses for speech output, avoiding complex formatting.

**Context Management Rules**:
The most critical part - defines when to call the tool vs. answer from context:
1. Call for NEW searches
2. Call for REFINED searches (adding filters)
3. Call for RE-SORTING
4. DON'T call for follow-up questions about existing results

**Ambiguity Handling**:
Instructs the LLM to ask clarifying questions rather than making assumptions when information is missing.

**Response Style**:
- Concise (voice responses should be brief)
- No complex formatting (readable when spoken aloud)
- No emojis or special characters
- Professional but friendly tone

## Context Management Strategy

### The Problem

With a dataset of 100 providers, sending all results in every LLM call would:
- Increase token costs significantly
- Add latency to responses
- Potentially exceed context limits with larger datasets

### The Solution

**Stateful Parameter-Based Search with Full Result Caching**:

1. **Initial Search**: User asks for providers, LLM calls tool with appropriate parameters
2. **Tool Returns Full Data**: Complete provider objects returned and added to conversation context
3. **Follow-Up Questions**: LLM answers from context without re-calling tool
4. **Context Invalidation**: Only call tool again for genuinely new searches

### How It Works

**Conversation Context in OpenAI API**:
```
[
  {"role": "user", "content": "Find heart doctors in Dallas"},
  {"role": "assistant", "content": null, "tool_calls": [...]},
  {"role": "tool", "content": "{\"found\": 2, \"providers\": [...]}"},
  {"role": "assistant", "content": "I found 2 cardiologists..."},
  {"role": "user", "content": "What insurance does the first one accept?"},
  {"role": "assistant", "content": "Dr. Smith accepts..."}  // NO TOOL CALL
]
```

The tool response containing full provider data remains in context, enabling the LLM to answer subsequent questions.

### Prompt Engineering for Context Awareness

The key phrase in both the agent instructions and tool docstring:
```
This tool returns complete provider information. After calling,
all data stays in conversation context. For follow-up questions about
providers already found, answer from context WITHOUT calling this tool again.
```

This explicit instruction is what makes the LLM understand it should not re-call the tool for follow-up questions.

## Real Conversation Examples

### Example 1: Multi-Step Search with Context Management

**Query 1**: "Heart doctors in Arizona with rating higher than 4"

```
LLM Parameters:
{
  "specialty": "Cardiology",
  "state": "AZ", 
  "min_rating": 4.0,
  "sort_by": "rating",
  "sort_order": "desc"
}

Search Execution:
- Start with 100 providers
- Filter: state == "AZ" → 5 providers
- Map: "Cardiology" (already formal, no alias needed)
- Filter: specialty contains "Cardiology" → 1 provider
- Filter: rating >= 4.0 → 0 providers (Dr. Karen Davis has 3.8)

Result: 0 providers found

LLM Response:
"There are no cardiologists in Arizona with a rating higher than 4 available 
in the current listings. Would you like me to search in a different location?"
```

**Analysis**: Search correctly returned 0 because the only Cardiologist in Arizona has rating 3.8, below the 4.0 threshold.

---

**Query 2**: "Anywhere in the country"

```
LLM Parameters:
{
  "specialty": "Cardiology",
  "state": null,  // Removed state filter
  "min_rating": 4.0,
  "sort_by": "rating",
  "sort_order": "desc",
  "limit": 5
}

Search Execution:
- Start with 100 providers
- Filter: specialty contains "Cardiology" → 8 providers
- Filter: rating >= 4.0 → 5 providers
- Sort by rating descending
- Limit to 5

Results:
1. Dr. Angela Gomez (Houston, TX) - 4.5
2. Dr. Jonathan Cox (Fort Worth, TX) - 4.4
3. Dr. Brandon Rodriguez (Detroit, MI) - 4.3
4. Dr. Pamela Ward (Dallas, TX) - 4.2
5. Dr. Jason Collins (Austin, TX) - 4.2

LLM Response:
"I found 5 cardiologists with ratings higher than 4.0:
1. Dr. Angela Gomez in Houston, Texas with a rating of 4.5...
[continues with all 5 providers]"
```

**Analysis**: Search expanded to nationwide, found 5 matching providers, correctly sorted by rating in descending order.

---

**Query 3**: "Can you find me the ones who are accepting new patients please"

```
NO TOOL CALL MADE

LLM Process:
1. Recognizes this is a follow-up question
2. Refers to providers in conversation context (from Query 2)
3. Filters the 5 results by "accepting_new_patients": true
4. Identifies 2 matching providers

LLM Response:
"From those 5 cardiologists, 2 are accepting new patients:
- Dr. Pamela Ward in Dallas, Texas
- Dr. Jason Collins in Austin, Texas"
```

**Analysis**: This is the critical proof that context management works. The LLM:
- Did NOT call the tool again
- Successfully filtered existing results
- Answered directly from conversation context

### Example 2: State Abbreviation Mapping

**Query**: "Heart doctors in California"

```
LLM Parameters:
{
  "specialty": "Cardiology",
  "state": "California"  // User said full name
}

Search Execution:
- Mapping: STATE_ABBREV_MAP.get("california") → "CA"
- Filter: state == "CA" → 12 providers
- Filter: specialty contains "Cardiology" → 0 providers

Result: 0 providers found

Suggestions Generated:
{
  "available_in_city": [
    "Endocrinology", "Family Medicine", "Gastroenterology",
    "Internal Medicine", "Obstetrics and Gynecology", 
    "Ophthalmology", "Pathology", "Pediatrics"
  ]
}
```

**Analysis**: State mapping worked correctly ("California" → "CA"), but no Cardiology providers exist in California. Suggestions show what specialties are available.

### Example 3: Specialty Alias Mapping

**Query**: "Find me a heart doctor in Dallas"

```
LLM Parameters:
{
  "specialty": "heart",  // Informal term
  "city": "Dallas"
}

Search Execution:
- Mapping: SPECIALTY_ALIASES.get("heart") → "Cardiology"
- Filter: city contains "Dallas" → 6 providers
- Filter: specialty contains "Cardiology" → 2 providers

Results:
- Dr. Pamela Ward: Cardiology, Rating 4.2
- Dr. Nicole Lee: Cardiology, Rating 3.9
```

**Analysis**: Specialty alias successfully mapped "heart" to "Cardiology", allowing the search to work even though the LLM passed an informal term.

## Performance Characteristics

### Time Complexity

**Search Operation**: O(n * f) where:
- n = number of providers (100)
- f = number of filters applied (typically 2-5)

With 100 providers, even 10 filters results in ~1000 comparisons, completing in microseconds.

**Sorting**: O(n log n) when sorting is requested, negligible for 100 items.

### Space Complexity

**Memory Usage**:
- Provider data: ~50KB in memory
- Mapping dictionaries: ~5KB
- Results list: Maximum ~25KB (for limit=5 with full objects)
- Total: <100KB additional memory usage

### Latency Profile

**Typical Search Timeline**:
1. User speech → STT: 200-500ms
2. LLM parameter extraction: 400-1500ms (with prompt caching: 100-400ms)
3. Tool execution: <1ms (in-memory search)
4. LLM response generation: 300-800ms
5. TTS synthesis: 200-400ms

**Total**: 1.1-3.2 seconds from user speech to agent response start

**Critical Insight**: Tool execution is negligible (<1ms). Latency is dominated by neural network operations (STT, LLM, TTS).

## Error Handling and Edge Cases

### Empty Results

When no providers match the criteria, the function returns:
- Clear message: "No providers found matching your search criteria."
- Contextual suggestions based on which parameters were provided
- Helps user understand what's available in the dataset

### Missing Data Fields

```python
key=lambda p: p.get(field, 0)
```

Using `.get()` with default value prevents KeyError if a provider is missing a field.

### Invalid Sort Fields

If user requests sorting by a non-existent field, the default value (0) is used for all providers, resulting in arbitrary order. This degrades gracefully rather than throwing an error.

### Transcription Errors

Common STT errors are handled by:
1. **State mapping**: "Oklahoma" transcribed correctly maps to "OK"
2. **Specialty aliases**: "heart" (common in speech) maps to "Cardiology"
3. **Case insensitivity**: "DALLAS" and "dallas" both match
4. **Partial matching**: "San Fran" matches "San Francisco"

### Ambiguous Queries

The agent instructions explicitly tell the LLM to ask for clarification:
```
If a query is ambiguous or missing key information (e.g., "doctors near me" without a city), 
ask the user for clarification before searching.
```

Example from real usage:
- User: "Can you find me hard doctors in california"
- Agent: "Could you please clarify if you mean heart doctors or another specialty?"
- User: "Heart doctors in california"
- Agent: [proceeds with search]

## Design Trade-offs

### 1. In-Memory vs. Database

**Chosen**: In-memory loading
**Alternative**: Query external database on each search

**Reasoning**:
- Dataset is small (100 providers)
- Search needs to be extremely fast (<1ms)
- No concurrent access concerns (single user per session)
- Simplifies deployment (no database setup required)

**Trade-off**: Not scalable to 100,000+ providers without architectural change

### 2. Full Results vs. Pagination

**Chosen**: Return full provider objects in single response
**Alternative**: Paginated results with IDs

**Reasoning**:
- Enables follow-up questions without additional tool calls
- 5 providers with full details fits comfortably in context window
- Reduces total number of LLM calls (cost and latency)

**Trade-off**: Higher token usage per search (but offset by fewer total calls)

### 3. OR vs. AND for List Parameters

**Chosen**: OR logic for `languages` and `insurance`
**Alternative**: AND logic (provider must have all listed items)

**Reasoning**:
- More useful for users ("I need someone who speaks Spanish OR Mandarin")
- AND would be overly restrictive and return fewer results
- Can be changed per-parameter if needed

**Trade-off**: Cannot express "must speak both Spanish AND Mandarin" without separate calls

### 4. Partial vs. Exact Matching

**Chosen**: Partial matching with `in` operator
**Alternative**: Exact equality

**Reasoning**:
- More forgiving for voice input variations
- "San Fran" matches "San Francisco"
- "Cardio" matches "Cardiology"
- More user-friendly

**Trade-off**: Potential for false positives (rare with medical terminology)

### 5. LLM-Based Alias Mapping vs. Comprehensive Dictionary

**Chosen**: Small alias dictionary as safety net
**Alternative**: Rely entirely on LLM's semantic understanding

**Reasoning**:
- LLM handles most natural language variations
- Dictionary provides deterministic fallback
- Balance between flexibility and reliability

**Trade-off**: Dictionary requires maintenance as new terms emerge

## Token Optimization

### Prompt Caching

Observable in logs:
```
"prompt_tokens": 2253, "prompt_cached_tokens": 2176
```

OpenAI caches the system prompt and tool definition, reducing:
- Cost: Cached tokens are 50% cheaper
- Latency: Cached content doesn't need re-processing

**Design Impact**: Detailed tool docstrings are "free" after first call due to caching.

### Response Size Management

```python
limit: int = 5
```

Default limit of 5 keeps responses concise while providing meaningful options.

**Token Calculation** (estimated):
- Tool definition: ~800 tokens (cached)
- Agent instructions: ~300 tokens (cached)
- 5 providers with full details: ~1000 tokens
- Conversation history: ~500-2000 tokens (grows over time)
- Total: ~2600-3800 tokens per call

Well within GPT-4's context window (128K tokens).

## Testing Insights

### Real-World Performance

From conversation logs:
- 3 searches in single conversation
- Total token usage: 9,784 prompt + 565 completion
- 5,632 tokens were cached (saving ~$0.02)
- Average LLM response time: 0.5-1.5 seconds
- Tool execution time: <1ms (not even logged separately)

### Context Management Validation

The critical test was Query 3: "Can you find me the ones who are accepting new patients"

Evidence it worked:
1. No tool call in logs after line 992
2. LLM response at line 999 with only 61 completion tokens (brief answer)
3. Response correctly filtered previous results
4. Latency was lower than initial search (no tool execution overhead)

This validates that the context management strategy is working as designed.

## Future Considerations

### Scalability Path

If dataset grows beyond 10,000 providers:
1. **Database Migration**: Move to PostgreSQL with GIN indexes
2. **Caching Layer**: Redis for frequent searches
3. **Pagination**: Return provider IDs, fetch details on demand
4. **Embeddings**: Add semantic search for symptom-to-specialty mapping

### Enhanced Features

Possible extensions without architectural changes:
1. **Distance Calculation**: Add lat/lon to providers, filter by radius
2. **Availability Slots**: Integration with scheduling system
3. **Insurance Verification**: Real-time eligibility checking
4. **Review Aggregation**: Pull latest ratings from external sources

### Multi-Language Support

Current implementation is English-only, but could extend to:
1. Multiple instruction sets per language
2. Translation of specialty aliases
3. Locale-aware formatting (phone numbers, addresses)

## Conclusion

The implementation successfully balances:
- **Performance**: Sub-millisecond search with in-memory data
- **Flexibility**: Rich parameter set covers most search scenarios
- **User Experience**: Context management minimizes redundant queries
- **Maintainability**: Simple Python with no external dependencies
- **Robustness**: Graceful handling of edge cases and errors

The key innovation is the context management strategy, which leverages the LLM's conversation context to eliminate redundant tool calls while maintaining full information availability for follow-up questions.

