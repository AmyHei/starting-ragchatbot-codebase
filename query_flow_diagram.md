# RAG System Query Processing Flow

```mermaid
graph TD
    %% Frontend Layer
    A[User Types Query] --> B[sendMessage() - script.js:45]
    B --> C[Disable UI & Show Loading]
    C --> D[POST /api/query]
    
    %% API Layer
    D --> E[FastAPI Endpoint - app.py:56]
    E --> F[Create/Get Session ID]
    F --> G[Call rag_system.query()]
    
    %% RAG System Layer
    G --> H[RAG System - rag_system.py:102]
    H --> I[Build AI Prompt]
    I --> J[Get Conversation History]
    J --> K[Call ai_generator.generate_response()]
    
    %% AI Generation Layer
    K --> L[Claude API - ai_generator.py]
    L --> M{Does query need<br/>course search?}
    
    %% Tool Usage Branch
    M -->|Yes| N[Use Search Tool]
    N --> O[CourseSearchTool]
    O --> P[VectorStore.search()]
    P --> Q[ChromaDB Query]
    Q --> R[Semantic Search Results]
    R --> S[Return Course Chunks]
    
    %% Direct Response Branch
    M -->|No| T[Use Claude Knowledge]
    
    %% Response Generation
    S --> U[Claude Processes Results]
    T --> U
    U --> V[Generate Final Response]
    
    %% Response Flow Back
    V --> W[Get Sources from Tool]
    W --> X[Update Session History]
    X --> Y[Return Response + Sources]
    Y --> Z[FastAPI Response JSON]
    
    %% Frontend Display
    Z --> AA[Remove Loading Animation]
    AA --> BB[addMessage() - Display Response]
    BB --> CC[Show Sources if Available]
    CC --> DD[Re-enable Input]
    
    %% Styling
    classDef frontend fill:#e1f5fe
    classDef api fill:#f3e5f5
    classDef rag fill:#e8f5e8
    classDef ai fill:#fff3e0
    classDef vector fill:#fce4ec
    classDef response fill:#f1f8e9
    
    class A,B,C,D,AA,BB,CC,DD frontend
    class E,F,Z api
    class G,H,I,J,W,X,Y rag
    class K,L,M,U,V ai
    class N,O,P,Q,R,S vector
    class T response
```

## Key Components & Data Flow

### 1. **Frontend (JavaScript)**
- **Input**: User query via chat interface
- **Processing**: Form validation, UI state management, HTTP request
- **Output**: POST request to `/api/query`

### 2. **API Layer (FastAPI)**
- **Input**: JSON with `{query, session_id}`
- **Processing**: Session management, RAG system orchestration
- **Output**: JSON with `{answer, sources, session_id}`

### 3. **RAG System (Python)**
- **Input**: Query string and session context
- **Processing**: Prompt building, history retrieval, AI coordination
- **Output**: Generated response with source attribution

### 4. **AI Generator (Claude)**
- **Input**: Prompt + conversation history + available tools
- **Decision**: Whether to search course materials or use general knowledge
- **Output**: Natural language response

### 5. **Vector Search (ChromaDB)**
- **Input**: Search query for course content
- **Processing**: Semantic similarity search across course chunks
- **Output**: Relevant course material chunks with metadata

### 6. **Response Assembly**
- Combines AI response with source attribution
- Updates conversation history for context
- Returns structured response to frontend

## Error Handling
- Network failures caught at frontend level
- API errors return HTTP status codes
- Search failures gracefully handled with fallback responses
- UI remains responsive with loading states