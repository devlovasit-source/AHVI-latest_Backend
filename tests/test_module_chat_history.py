import pytest
from pydantic import ValidationError

from routers.chat import ModuleChatRequest


def _history(count):
    history = [
        {"role": "assistant", "content": f"message-{index}"}
        for index in range(count)
    ]
    if history:
        history[-1] = {"role": "user", "content": f"latest-user-{count}"}
    return history


@pytest.mark.parametrize("count", [19, 20, 21, 50, 100])
def test_module_chat_history_keeps_newest_twenty(count):
    history = _history(count)

    request = ModuleChatRequest(message="Continue", history=history)

    assert request.history == history[-20:]
    assert len(request.history) == min(count, 20)


def test_long_history_preserves_order_and_latest_user_while_removing_oldest():
    history = _history(50)

    request = ModuleChatRequest(message="Continue", history=history)

    assert request.history == history[30:]
    assert request.history[0]["content"] == "message-30"
    assert request.history[-1] == {"role": "user", "content": "latest-user-50"}
    assert history[0] not in request.history


def test_malformed_non_list_history_is_rejected():
    with pytest.raises(ValidationError):
        ModuleChatRequest(
            message="Continue",
            history={"role": "user", "content": "not-a-list"},
        )


def test_history_bounding_does_not_change_context_or_board_identity():
    context = {"occasion": "office", "nested": {"weather": "warm"}}
    style_state = {"board_id": "board-123", "revision": 7, "locked_slots": ["top"]}

    request = ModuleChatRequest(
        message="Continue",
        history=_history(100),
        context=context,
        style_state=style_state,
        current_look_id="look-456",
    )

    assert request.context == context
    assert request.style_state == style_state
    assert request.style_state["board_id"] == "board-123"
    assert request.current_look_id == "look-456"
