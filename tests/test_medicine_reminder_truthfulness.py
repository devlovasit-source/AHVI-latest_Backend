import asyncio

from services.module_chat_service import handle_medi_chat


def test_medicine_reminder_chat_does_not_claim_unsaved_schedule():
    response = asyncio.run(
        handle_medi_chat(
            "Remind me to take my medicine at 8 PM",
            {"medications": []},
            "user-1",
        )
    )

    assert response["intent"] == "medicine_reminder_not_scheduled"
    assert response["persisted"] is False
    assert response["data"]["intent"] == "medicine_reminder_not_scheduled"
    assert response["data"]["persisted"] is False
    assert response["chips"] == ["Open Medicines"]
    message = response["message_text"].lower()
    assert "no reminder was saved" in message
    assert "scheduled" not in message
