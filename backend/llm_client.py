"""
NEURON v2.0 LLM Client
Multi-provider LLM integration (Ollama, NVIDIA, Anthropic, OpenAI, SiliconFlow)
"""

from typing import Optional, List, Dict, Any
import os
import json
import httpx
from abc import ABC, abstractmethod

# Provider configurations
PROVIDERS = {
    "ollama": {
        "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        "model": os.environ.get("OLLAMA_MODEL", "qwen3:8b"),
    },
    "nvidia": {
        "base_url": "https://api.nvcf.nvidia.com/v1",
        "api_key": os.environ.get("NVIDIA_API_KEY", ""),
        "model": os.environ.get("NVIDIA_MODEL", "qwen/qwen3.5-397b-a17b"),
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o"),
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "api_key": os.environ.get("SILICON_API_KEY", ""),
        "model": os.environ.get("SILICON_MODEL", "deepseek-ai/DeepSeek-V3"),
    }
}

# Default provider
DEFAULT_PROVIDER = os.environ.get("NEURON_DEFAULT_PROVIDER", "ollama")


class LLMClient(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], system_prompt: str = None) -> str:
        pass
    
    @abstractmethod
    async def learn(self, user_input: str, context: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        pass


class OllamaClient(LLMClient):
    """Ollama local LLM client"""
    
    def __init__(self, model: str = None, base_url: str = None):
        self.model = model or PROVIDERS["ollama"]["model"]
        self.base_url = base_url or PROVIDERS["ollama"]["base_url"]
    
    async def chat(self, messages: List[Dict[str, str]], system_prompt: str = None) -> str:
        """Chat completion"""
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Build messages array
            all_messages = []
            if system_prompt:
                all_messages.append({"role": "system", "content": system_prompt})
            all_messages.extend(messages)
            
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": all_messages,
                    "stream": False
                }
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
    
    async def learn(self, user_input: str, context: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """Process learning with structured output"""
        system_prompt = self._build_learning_prompt(strategy, context)
        
        messages = [{"role": "user", "content": user_input}]
        response = await self.chat(messages, system_prompt)
        
        return self._parse_learning_response(response)
    
    def _build_learning_prompt(self, strategy: str, context: Dict[str, Any]) -> str:
        """Build system prompt for learning"""
        strategy_prompts = {
            "CoT": "Reason step by step: [step1] → [step2] → [step3]",
            "ToT": "Branch into 2-3 hypotheses, evaluate each, select strongest",
            "Synthesis": "Cross-reference with existing knowledge to find non-obvious connections",
            "Socratic": "Guide via questions that lead to deeper understanding",
            "Analysis": "Deconstruct into first principles, then rebuild understanding"
        }
        
        return f"""You are NEURON v2.0, a self-learning AI agent.
Your active learning strategy: {strategy}
Strategy guide: {strategy_prompts.get(strategy, "Think carefully")}

Knowledge base: {context.get('kb_count', 0)} entries
Concepts: {context.get('concepts', [])}
Goals: {context.get('goals', [])}

Respond ONLY with valid JSON in this format:
{{
    "response": "Your conversational answer",
    "reasoning": ["step 1", "step 2", "step 3"],
    "learned": {{
        "concepts": ["concept1", "concept2"],
        "patterns": ["pattern1"],
        "domain": "Science|Technology|Philosophy|Arts|History|Math|Language|Psychology|General",
        "subDomain": "specific area",
        "confidence": 0.8,
        "reliability": 0.7,
        "keyInsight": "the main insight",
        "secondaryInsight": "secondary insight",
        "complexity": "basic|intermediate|advanced|expert",
        "hypotheses": ["testable hypothesis"],
        "curiosityQuestions": ["question1"],
        "crossDomainLinks": [{{"domain": "OtherDomain", "connection": "how they connect", "novelty": 0.7}}],
        "mnemonicHook": "memory anchor phrase"
    }}
}}"""
    
    def _parse_learning_response(self, response: str) -> Dict[str, Any]:
        """Parse learning response with robust error handling"""
        try:
            # Try to extract JSON from response
            import re
            
            # Remove code block markers if present
            clean_response = response.strip()
            clean_response = re.sub(r'^```json\s*', '', clean_response)
            clean_response = re.sub(r'\s*```$', '', clean_response)
            clean_response = re.sub(r'^```\s*', '', clean_response)
            
            # Try to find JSON object
            json_match = re.search(r'\{[\s\S]*\}', clean_response)
            if json_match:
                json_str = json_match.group(0)
                
                # Try direct parsing
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Try to fix common JSON errors
                    # Fix double braces
                    json_str = json_str.replace('{{', '{').replace('}}', '}')
                    # Remove trailing commas
                    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                    
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        # Try even more aggressive cleanup
                        print(f"JSON parse error: {e}")
                        # Return what we can extract manually
                        return {
                            "response": response,
                            "learned": self._extract_basic_learning(response)
                        }
            
            return {"response": response, "learned": None}
        except Exception as e:
            print(f"Error parsing learning response: {e}")
            return {"response": response, "learned": None}
    
    def _extract_basic_learning(self, response: str) -> Dict[str, Any]:
        """Extract basic learning info from raw response"""
        # Simple extraction if JSON parsing fails
        return {
            "concepts": [],
            "domain": "General",
            "confidence": 0.5,
            "reliability": 0.5,
            "keyInsight": response[:200] if response else "Extracted from response",
            "complexity": "basic"
        }


class NVIDIAClient(LLMClient):
    """NVIDIA API client"""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or PROVIDERS["nvidia"]["api_key"]
        self.model = model or PROVIDERS["nvidia"]["model"]
        self.base_url = PROVIDERS["nvidia"]["base_url"]
    
    async def chat(self, messages: List[Dict[str, str]], system_prompt: str = None) -> str:
        """Chat completion via NVIDIA API"""
        async with httpx.AsyncClient(timeout=120.0) as client:
            all_messages = []
            if system_prompt:
                all_messages.append({"role": "system", "content": system_prompt})
            all_messages.extend(messages)
            
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": all_messages,
                    "max_tokens": 2000,
                    "temperature": 0.7
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    
    async def learn(self, user_input: str, context: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """Process learning with structured output"""
        # Similar to OllamaClient but with NVIDIA-specific handling
        client = OllamaClient()
        return await client.learn(user_input, context, strategy)


class AnthropicClient(LLMClient):
    """Anthropic API client"""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or PROVIDERS["anthropic"]["api_key"]
        self.model = model or PROVIDERS["anthropic"]["model"]
        self.base_url = PROVIDERS["anthropic"]["base_url"]
    
    async def chat(self, messages: List[Dict[str, str]], system_prompt: str = None) -> str:
        """Chat completion via Anthropic API"""
        async with httpx.AsyncClient(timeout=120.0) as client:
            all_messages = []
            if system_prompt:
                all_messages.append({"role": "user", "content": f"\n\nSystem: {system_prompt}"})
            all_messages.extend(messages)
            
            response = await client.post(
                f"{self.base_url}/messages",
                headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
                json={
                    "model": self.model,
                    "messages": all_messages,
                    "max_tokens": 2000
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["content"][0]["text"]
    
    async def learn(self, user_input: str, context: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """Process learning with structured output"""
        client = OllamaClient()
        return await client.learn(user_input, context, strategy)


class OpenAIClient(LLMClient):
    """OpenAI API client"""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or PROVIDERS["openai"]["api_key"]
        self.model = model or PROVIDERS["openai"]["model"]
        self.base_url = PROVIDERS["openai"]["base_url"]
    
    async def chat(self, messages: List[Dict[str, str]], system_prompt: str = None) -> str:
        """Chat completion via OpenAI"""
        async with httpx.AsyncClient(timeout=120.0) as client:
            all_messages = []
            if system_prompt:
                all_messages.append({"role": "system", "content": system_prompt})
            all_messages.extend(messages)
            
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": all_messages,
                    "max_tokens": 2000,
                    "temperature": 0.7
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    
    async def learn(self, user_input: str, context: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """Process learning with structured output"""
        client = OllamaClient()
        return await client.learn(user_input, context, strategy)


class SiliconFlowClient(LLMClient):
    """SiliconFlow API client"""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or PROVIDERS["siliconflow"]["api_key"]
        self.model = model or PROVIDERS["siliconflow"]["model"]
        self.base_url = PROVIDERS["siliconflow"]["base_url"]
    
    async def chat(self, messages: List[Dict[str, str]], system_prompt: str = None) -> str:
        """Chat completion via SiliconFlow"""
        async with httpx.AsyncClient(timeout=120.0) as client:
            all_messages = []
            if system_prompt:
                all_messages.append({"role": "system", "content": system_prompt})
            all_messages.extend(messages)
            
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": all_messages,
                    "max_tokens": 2000,
                    "temperature": 0.7
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    
    async def learn(self, user_input: str, context: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """Process learning with structured output"""
        client = OllamaClient()
        return await client.learn(user_input, context, strategy)


def get_llm_client(provider: str = None) -> LLMClient:
    """Factory function to get LLM client"""
    provider = provider or DEFAULT_PROVIDER
    
    clients = {
        "ollama": OllamaClient,
        "nvidia": NVIDIAClient,
        "anthropic": AnthropicClient,
        "openai": OpenAIClient,
        "siliconflow": SiliconFlowClient,
    }
    
    client_class = clients.get(provider, OllamaClient)
    return client_class()


async def test_llm_connection(provider: str = None) -> Dict[str, Any]:
    """Test LLM connection"""
    client = get_llm_client(provider)
    try:
        response = await client.chat(
            [{"role": "user", "content": "Say hello"}],
            "You are a helpful assistant."
        )
        return {"status": "success", "response": response, "provider": provider}
    except Exception as e:
        return {"status": "error", "error": str(e), "provider": provider}
