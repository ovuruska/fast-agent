import asyncio
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.prompt import Prompt

_ = load_dotenv(find_dotenv())

# Create the application
fast = FastAgent("fast-agent example")


# Define the agent
@fast.agent(instruction="You are a helpful AI Agent", servers=["filesystem", "fetch"])
async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:
        """
        await agent.default.generate(
            [
                Prompt.user(
                    Path("cat.png"), "Write a report on the content of the image to 'report.md'. Use markdown format."
                )
            ]
        )
        """
        await agent.default.generate(
            [
                Prompt.user(
                    "Write a random report on the content about US elections to 'report.md'. Use markdown format."
                )
            ]
        )
        await agent.interactive()


if __name__ == "__main__":
    asyncio.run(main())
