# Topic Extraction Network Error Fix

## Problem Description

Users were experiencing "Error extracting topics" and "Network error. Please check your connection." messages when trying to extract topics from PDF materials. This was caused by several issues:

1. **GROQ API Configuration**: Missing or invalid GROQ API key
2. **Network Connectivity**: API calls failing due to network issues or timeouts
3. **Poor Error Handling**: Generic error messages without specific details
4. **Missing Fallback Mechanisms**: No graceful degradation when AI service is unavailable

## Root Cause Analysis

The error was occurring in the following flow:
1. Frontend calls `/api/chapter-materials/extract-topics`
2. Backend uses `AIContentAnalyzer` which requires GROQ API
3. If GROQ API is not configured or network fails, the entire process fails
4. Frontend receives generic error and shows "Network error" message

## Solutions Implemented

### 1. Improved Error Handling in AI Service

**File**: `app/utils/ai_service.py`

- Added detailed error categorization (network, authentication, general)
- Added timeout configurations for API calls
- Improved initialization error reporting
- Better fallback mechanisms when AI service is unavailable
- **NEW**: Implemented chunked processing for large PDFs to prevent timeouts
- **NEW**: Added progressive content reduction based on size
- **NEW**: Shorter timeouts per chunk (30s) instead of single long timeout (120s)

### 2. Enhanced Backend Error Logging

**File**: `app/routers/chapter_materials.py`

- Added detailed logging for AI analysis failures
- Better error message propagation to frontend
- Improved fallback topic generation

### 3. System Status Endpoint

**File**: `app/routers/system_status.py` (NEW)

- Added `/api/system/ai-status` endpoint to check AI service health
- Added `/api/system/configuration-help` endpoint with setup instructions
- Allows admins to diagnose configuration issues

### 4. Better Frontend Error Handling

**File**: `frontend/src/pages/member/SelectedMaterials.jsx`

- More specific error messages based on error type
- Better error logging for debugging
- Improved user feedback
- **NEW**: Increased timeout from 2 minutes to 5 minutes
- **NEW**: Added progress feedback with 30-second updates
- **NEW**: Better loading messages for long operations

### 5. Configuration Documentation

**File**: `.env.example`

- Already contains GROQ API configuration examples
- Clear instructions for setting up environment variables

## Setup Instructions

### Step 1: Get GROQ API Key

1. Visit [GROQ Console](https://console.groq.com/)
2. Sign up for an account
3. Create a new API key
4. Copy the API key

### Step 2: Configure Environment

Create a `.env` file in the backend root directory:

```bash
# Copy from .env.example
cp .env.example .env
```

Edit `.env` and set your GROQ API key:

```env
GROQ_API_KEY=your_actual_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

### Step 3: Restart Application

Restart your FastAPI server to load the new environment variables:

```bash
# If running with uvicorn
uvicorn app.main:app --reload

# If running with python
python -m app.main
```

### Step 4: Verify Configuration

Check if the AI service is working:

1. Login as an admin user
2. Visit: `GET /api/system/ai-status`
3. Check the response for any configuration issues

Example successful response:
```json
{
  "status": true,
  "message": "AI service is operational",
  "details": {
    "environment": {
      "groq_api_key_set": true,
      "groq_api_key_valid": true,
      "groq_model": "llama-3.3-70b-versatile"
    },
    "ai_service": {
      "api_available": true,
      "initialization_error": null,
      "model": "llama-3.3-70b-versatile"
    },
    "test_result": {
      "test_successful": true,
      "sections_found": 1,
      "test_content_length": 85
    }
  }
}
```

## Troubleshooting

### Common Issues

1. **"GROQ_API_KEY environment variable is not set"**
   - Solution: Set the GROQ_API_KEY in your .env file
   - Restart the application after setting

2. **"Network connectivity issue"**
   - Check your internet connection
   - Verify firewall settings allow outbound HTTPS connections
   - Check if your network blocks AI service domains

3. **"API authentication failed"**
   - Verify your GROQ API key is correct
   - Check if your API key has sufficient credits
   - Ensure the API key hasn't expired

4. **"Timeout errors"**
   - Check network latency to GROQ servers
   - Consider increasing timeout values if needed
   - Verify stable internet connection

### Fallback Behavior

When AI service is unavailable, the system will:

1. Use rule-based topic extraction
2. Generate language-appropriate fallback topics
3. Provide basic content sections
4. Log detailed error information for debugging

### Testing the Fix

1. **Test with AI service available**:
   - Upload a PDF material
   - Click "Extract Topics"
   - Should see extracted topics in detected language
   - **NEW**: Large PDFs should process in chunks without timeout

2. **Test with AI service unavailable**:
   - Temporarily set invalid GROQ_API_KEY
   - Try extracting topics
   - Should see fallback topics and appropriate error messages

3. **Test network error handling**:
   - Disconnect internet temporarily
   - Try extracting topics
   - Should see specific network error message

4. **Test chunked processing**:
   - Use the test endpoint: `POST /api/system/test-topic-extraction`
   - Should complete within reasonable time
   - Check processing time in response

5. **Test timeout handling**:
   - Try with very large PDF (>50MB)
   - Should process in chunks with progress updates
   - Should not exceed 5-minute frontend timeout

## Monitoring

Use the system status endpoint to monitor AI service health:

```bash
# Check AI service status
curl -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
     http://localhost:8014/api/system/ai-status

# Get configuration help
curl -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
     http://localhost:8014/api/system/configuration-help

# Test topic extraction
curl -X POST -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{}' \
     http://localhost:8014/api/system/test-topic-extraction
```

## Files Modified

1. `app/utils/ai_service.py` - Improved error handling and timeouts
2. `app/routers/chapter_materials.py` - Better error logging
3. `app/routers/system_status.py` - NEW: System diagnostics
4. `app/main.py` - Added system status router
5. `frontend/src/pages/member/SelectedMaterials.jsx` - Better error messages

## Additional Notes

- The system now gracefully handles AI service unavailability
- Fallback topics are generated in the appropriate language
- Detailed logging helps with debugging
- System status endpoint helps with configuration verification
- All changes are backward compatible
