"""
Compact LLMLocalRouter - FastAPI service for LLM provider routing with intelligent failover
Enhanced with comprehensive exponential backoff retry logic (10s to 10min range)
"""

import asyncio, json, time, os
from typing import Dict, List, Any
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# Import enhanced retry utilities
from src.utils.retry_utils import EnhancedRetryHandler, EnhancedRetryConfig, RetryErrorType

class Provider:
    def __init__(self, name: str, cfg: Dict):
        self.name, self.url, self.key, self.model = name, cfg["url"], cfg.get("key", cfg.get("api", "")), cfg["model"]
        self.max_ctx = cfg.get("max_context_size", 4000)
        self.last_fail, self.failures = 0, 0
        
        # Initialize enhanced retry handler for this provider
        self.retry_handler = EnhancedRetryHandler(EnhancedRetryConfig(
            max_attempts=5,
            base_delay=10.0,  # 10 seconds minimum
            max_delay=600.0,  # 10 minutes maximum
            rate_limit_multiplier=3.0,  # Higher multiplier for rate limits
            network_multiplier=2.0
        ))
    
    def available(self) -> bool:
        if self.failures == 0: return True
        # Enhanced exponential backoff: 10s * 2^failures, capped at 10 minutes
        backoff = min(600, 10 * (2 ** min(self.failures - 1, 6)))
        return time.time() - self.last_fail > backoff
    
    def success(self):
        self.failures = 0;
        self.last_fail = 0
        # Reset retry stats on success
        self.retry_handler.reset_stats()
    
    def failure(self):
        self.failures += 1;
        self.last_fail = time.time()
    
    def get_retry_stats(self) -> Dict[str, Any]:
        """Get retry statistics for this provider"""
        stats = self.retry_handler.get_stats()
        # Convert RetryErrorType enum to string for JSON serialization
        if 'last_error_type' in stats and stats['last_error_type'] is not None:
            stats['last_error_type'] = stats['last_error_type'].value
        
        # Convert error_counts dict keys from RetryErrorType to strings
        if 'error_counts' in stats and isinstance(stats['error_counts'], dict):
            error_counts = {}
            for error_type, count in stats['error_counts'].items():
                if hasattr(error_type, 'value'):
                    error_counts[error_type.value] = count
                else:
                    error_counts[str(error_type)] = count
            stats['error_counts'] = error_counts
        
        return stats

class Router:
    def __init__(self):
        self.providers, self.client = [], httpx.AsyncClient(timeout=30.0)
        try:
            path = os.path.expanduser("~/.conjecture/config.json")
            with open(path) as f: cfg = json.load(f)
            for name, pc in cfg.get("providers", {}).items(): self.providers.append(Provider(name, pc))
        except: self.providers = [Provider("default", {"url": "http://localhost:1234", "model": "local"})]
    
    def estimate_tokens(self, msgs: List[Dict]) -> int:
        return sum(len(str(m.get("content", ""))) for m in msgs) // 4
    
    async def get_models(self):
        return {"object": "list", "data": [
            {"id": f"{p.name}:{p.model}", "object": "model", "created": int(time.time()),
             "owned_by": p.name, "provider": p.name, "url": p.url, "max_context_size": p.max_ctx}
            for p in self.providers if p.available()
        ]}
    
    async def forward(self, data: Dict, model: str = None):
        providers = [p for p in self.providers if not model or not ":" in model or p.name == model.split(":")[0]]
        tokens = self.estimate_tokens(data.get("messages", []))
        last_err = None
        
        for p in providers:
            if not p.available() or tokens > p.max_ctx:
                print(f"Skipping provider {p.name}: available={p.available()}, tokens={tokens}, max_ctx={p.max_ctx}")
                continue
            
            # Use enhanced retry logic for each provider
            try:
                result = await p.retry_handler.execute_with_retry_async(
                    self._call_provider, p, data
                )
                p.success()
                result["provider"] = p.name
                result["retry_stats"] = p.get_retry_stats()
                print(f"  SUCCESS with {p.name}: {result.get('id', 'N/A')}")
                return result
                
            except Exception as e:
                print(f"  All retry attempts failed for {p.name}: {e}")
                last_err = str(e)
                p.failure()
        
        raise HTTPException(503, f"All providers failed: {last_err}")
    
    async def _call_provider(self, provider: Provider, data: Dict) -> Dict[str, Any]:
        """Make API call to a specific provider with enhanced error handling"""
        headers = {"Content-Type": "application/json", **({} if not provider.key else {"Authorization": f"Bearer {provider.key}"})}
        req = {"model": provider.model, "messages": data.get("messages", []),
               "max_tokens": data.get("max_tokens", 1000), "temperature": data.get("temperature", 0.7)}
        url = provider.url.rstrip('/')
        
        # Special handling for different API endpoints
        if "api.z.ai" in url and url.endswith('/v4'):
            url += "/chat/completions"
        elif not url.endswith('/chat/completions'):
            url += "/v1/chat/completions" if not url.endswith('/v1') else "/chat/completions"
        
        print(f"  Attempting call to {provider.name}:")
        print(f"    URL: {url}")
        print(f"    Model: {req['model']}")
        
        resp = await self.client.post(url, json=req, headers=headers)
        resp.raise_for_status()
        return resp.json()

router = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global router
    router = Router()
    yield
    await router.client.aclose()

app = FastAPI(title="LLMLocalRouter", lifespan=lifespan)

@app.get("/v1/models")
async def list_models(): return await router.get_models()

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    return JSONResponse(await router.forward(data, data.get("model")))

@app.get("/health")
async def health():
    available = sum(1 for p in router.providers if p.available())
    return {"status": "healthy" if available > 0 else "unhealthy", "available": available, "total": len(router.providers)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5677)