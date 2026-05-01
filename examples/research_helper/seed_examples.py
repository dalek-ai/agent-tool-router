"""Seed examples for Router.from_examples().

Each entry is (natural-language task, [tool names]). With ~60 examples across
5 tools we get a usable router. With more you get a better one — same shape.

Multi-tool examples (mixed routes) are intentional: many real tasks chain a
search and a computation, or a memory recall and a file read. The router
returns top-k, so for those you just check `top-2` instead of `top-1`.
"""

SEED_EXAMPLES: list[tuple[str, list[str]]] = [
    # web_search (15)
    ("search the web for recent news on AI agents", ["web_search"]),
    ("look up information about Python online", ["web_search"]),
    ("find articles about quantum computing", ["web_search"]),
    ("what does the internet say about transformer architectures", ["web_search"]),
    ("find me the latest paper on diffusion models", ["web_search"]),
    ("google for the current price of bitcoin", ["web_search"]),
    ("search online for benchmarks of new LLMs", ["web_search"]),
    ("look up the wikipedia entry for graph neural networks", ["web_search"]),
    ("find recent blog posts about agent tool use", ["web_search"]),
    ("what is the consensus online on attention sinks", ["web_search"]),
    ("search for tutorials on retrieval augmented generation", ["web_search"]),
    ("find news from today about openai", ["web_search"]),
    ("look up online reviews for the latest macbook", ["web_search"]),
    ("search for conference deadlines in machine learning", ["web_search"]),
    ("find me online references on speculative decoding", ["web_search"]),

    # calculator (12)
    ("compute the square root of 144", ["calculator"]),
    ("what is 25 times 17", ["calculator"]),
    ("add 142 and 358 and divide by 4", ["calculator"]),
    ("evaluate 3.14 * 2 * 7", ["calculator"]),
    ("how much is 432 divided by 8", ["calculator"]),
    ("calculate 2 plus 2", ["calculator"]),
    ("what is 1024 minus 768", ["calculator"]),
    ("compute the percentage 47 of 230", ["calculator"]),
    ("multiply 12 by 13 by 14", ["calculator"]),
    ("how many seconds in 3 hours and 22 minutes", ["calculator"]),
    ("evaluate this arithmetic expression: 7 * 9 - 3", ["calculator"]),
    ("what is 2 to the power of 10", ["calculator"]),

    # file_read (12)
    ("read the contents of report.pdf", ["file_read"]),
    ("show me what is in the config file", ["file_read"]),
    ("open ./notes.md and summarize", ["file_read"]),
    ("what is in /tmp/log.txt", ["file_read"]),
    ("read the README from this project", ["file_read"]),
    ("display the contents of data.csv", ["file_read"]),
    ("open the local file requirements.txt", ["file_read"]),
    ("cat the file /etc/hosts", ["file_read"]),
    ("read the json file in ./data/", ["file_read"]),
    ("show me the first lines of the source file", ["file_read"]),
    ("load the local file and print it", ["file_read"]),
    ("get the contents of my notes file", ["file_read"]),

    # memory_lookup (12)
    ("what did the user say earlier about deadlines", ["memory_lookup"]),
    ("recall the meeting notes from earlier in this conversation", ["memory_lookup"]),
    ("what was decided about pricing", ["memory_lookup"]),
    ("remind me what i said about the budget earlier", ["memory_lookup"]),
    ("what did we discuss about the launch", ["memory_lookup"]),
    ("did i mention anything about the q3 review", ["memory_lookup"]),
    ("what was the conclusion from earlier", ["memory_lookup"]),
    ("recall what we agreed on about the api design", ["memory_lookup"]),
    ("what was that name i mentioned a minute ago", ["memory_lookup"]),
    ("remind me of the key point i made earlier", ["memory_lookup"]),
    ("what did we say about the rollout plan", ["memory_lookup"]),
    ("remember when i talked about the migration plan", ["memory_lookup"]),

    # python_exec (12)
    ("run a python script to plot a histogram", ["python_exec"]),
    ("execute some pandas code on this dataframe", ["python_exec"]),
    ("write a quick script to parse this CSV", ["python_exec"]),
    ("run python to compute fibonacci of 30", ["python_exec"]),
    ("execute this snippet of python code", ["python_exec"]),
    ("run a small program that loops over a list", ["python_exec"]),
    ("write a python function and run it on these inputs", ["python_exec"]),
    ("execute pandas groupby on the dataframe", ["python_exec"]),
    ("run a python regex to extract emails", ["python_exec"]),
    ("execute a quick numpy computation", ["python_exec"]),
    ("run python code to read a json and pretty print", ["python_exec"]),
    ("write and run python to merge two dicts", ["python_exec"]),

    # multi-tool (5)
    ("search online for current bitcoin price then convert it to euros", ["web_search", "calculator"]),
    ("read the csv file and write python to compute the mean", ["file_read", "python_exec"]),
    ("look up the population of france and divide by 4", ["web_search", "calculator"]),
    ("recall the budget i mentioned and divide it by twelve months", ["memory_lookup", "calculator"]),
    ("read the local notes and search online for related papers", ["file_read", "web_search"]),
]
