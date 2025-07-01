from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    async def run(self, *args, **kwargs):
        """Execute the agent's main logic."""
        pass