from customer_agent.evaluation.user_simulator import ScriptedSingleTurn, get_default_simulator


def test_single_turn_asks_dataset_question():
    sim = ScriptedSingleTurn()
    row = {"question": "How do I X?", "answer": "Do Y."}
    assert sim.first_message(row) == "How do I X?"


def test_single_turn_never_follows_up():
    sim = ScriptedSingleTurn()
    row = {"question": "How do I X?", "answer": "Do Y."}
    assert sim.next_message(row, "agent said something", turn=1) is None


def test_default_simulator_is_single_turn():
    assert isinstance(get_default_simulator(), ScriptedSingleTurn)
