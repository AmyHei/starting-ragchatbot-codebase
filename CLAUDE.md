# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start (recommended)
chmod +x run.sh && ./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Setup Commands
```bash
# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Testing & Development
```bash
# Access application
# Web interface: http://localhost:8000
# API docs: http://localhost:8000/docs

# Check ChromaDB data
# Database stored in: ./chroma_db/
```

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for course materials with a Python backend and vanilla JavaScript frontend.

### Core Data Flow

1. **Document Processing Pipeline**: 
   - Raw course documents (`/docs/*.txt`) → `DocumentProcessor` → structured `Course` objects with `Lesson` hierarchy
   - Text chunked into 800-character segments with 100-character overlap for semantic search
   - Each chunk enhanced with course/lesson context before storage

2. **Vector Storage Architecture**:
   - **Two ChromaDB collections**: `course_catalog` (metadata) + `course_content` (chunks)
   - `course_catalog`: Stores course titles, instructors, lesson metadata for course discovery
   - `course_content`: Stores text chunks with embeddings for semantic search
   - Uses SentenceTransformer embeddings (`all-MiniLM-L6-v2`)

3. **Query Processing Flow**:
   - Frontend → FastAPI → RAG System → AI Generator (Claude) → optional Vector Search → Response
   - Claude intelligently decides whether to search course materials or use general knowledge
   - Session management maintains conversation context across queries

### Key Components Integration

- **`RAGSystem`**: Main orchestrator coordinating all components
- **`AIGenerator`**: Claude API integration with tool-based search capabilities  
- **`VectorStore`**: ChromaDB wrapper handling dual-collection storage and semantic search
- **`DocumentProcessor`**: Converts structured course documents into searchable chunks
- **`SessionManager`**: Maintains conversation history for contextual responses
- **`ToolManager + CourseSearchTool`**: Enables Claude to search course materials when needed

### Document Format Requirements

Course documents must follow this structure:
```
Course Title: [title]
Course Link: [optional URL]
Course Instructor: [optional instructor]

Lesson 0: [lesson title]
Lesson Link: [optional URL]
[lesson content...]

Lesson 1: [next lesson title]
[content continues...]
```

### Configuration

Key settings in `backend/config.py`:
- `CHUNK_SIZE: 800` - Character limit for text chunks
- `CHUNK_OVERLAP: 100` - Overlap between consecutive chunks
- `MAX_RESULTS: 5` - Maximum search results returned
- `ANTHROPIC_MODEL: "claude-sonnet-4-20250514"`
- `EMBEDDING_MODEL: "all-MiniLM-L6-v2"`

### API Endpoints

- `POST /api/query` - Process user queries with RAG system
- `GET /api/courses` - Get course catalog statistics
- `GET /` - Serve frontend static files
- `GET /docs` - API documentation

### Data Persistence

- ChromaDB data: `./chroma_db/` directory
- Course documents: `./docs/` directory  
- Documents auto-loaded on application startup
- Duplicate course detection prevents re-processing existing content
- The vector database has two collections:
course_catalog:
stores course titles for name resolution
metadata for each course: title, instructor, course_link, lesson_count, lessons_json (list of lessons: lesson_number, lesson_title, lesson_link)
course_content:
stores text chunks for semantic search
metadata for each chunk: course_title, lesson_number, chunk_index