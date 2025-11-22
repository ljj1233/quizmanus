def create_react_agent(llm, tools=None, prompt=None):
    def agent(state=None):
        return {"llm": llm, "tools": tools or [], "prompt": prompt, "state": state}
    return agent
