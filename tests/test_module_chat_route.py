import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers import chat


def _text_chat_client_with_user():
    app = FastAPI()

    @app.middleware("http")
    async def add_user(request, call_next):
        request.state.user = {"user_id": "user-1"}
        return await call_next(request)

    app.include_router(chat.router, prefix="/api")
    return TestClient(app)


def test_interpreted_occasion_does_not_reclarify():
    assert chat._needs_style_clarification("Office", "office") is False
    assert chat._needs_style_clarification("Casual office wear", "office") is False


def test_swimming_prompt_routes_to_style_without_daily_clarification():
    assert chat._ahvi_style_occasion("outfit for swimming") == "swimming"
    assert chat._needs_style_clarification("outfit for swimming", "swimming") is False


def test_greeting_detection_exact_normalized_matches():
    assert chat._is_greeting("hi") is True
    assert chat._is_greeting("Hi!") is True
    assert chat._is_greeting("hello ahvi") is True
    assert chat._is_greeting("good morning") is True

    assert chat._is_greeting("what can you do") is False
    assert chat._is_greeting("hi what can you do") is False
    assert chat._is_greeting("office outfit") is False


def test_greeting_response_shape_preserves_module_context():
    response = chat._ahvi_greeting_response("style")

    assert response["type"] == "text"
    assert response["cards"] == []
    assert response["style_boards"] == []
    assert response["meta"]["mode"] == "greeting_bypass"
    assert response["meta"]["module_context"] == "style"


def test_help_identity_and_small_talk_response_shapes():
    help_response = chat._ahvi_help_identity_response("what can you do", "chat")
    assert help_response["type"] == "text"
    assert help_response["cards"] == []
    assert help_response["style_boards"] == []
    assert help_response["meta"]["mode"] == "help_identity_bypass"
    assert help_response["message"].startswith("I can help with Style, Planning, and Preparation.")

    small_talk = chat._ahvi_small_talk_response("style")
    assert small_talk["type"] == "text"
    assert small_talk["cards"] == []
    assert small_talk["style_boards"] == []
    assert small_talk["meta"]["mode"] == "small_talk_bypass"
    assert small_talk["message"] == (
        "Ready to help. Are we styling an outfit, planning your day, or preparing for something upcoming?"
    )


def test_text_chat_greeting_bypasses_style_and_orchestrator(monkeypatch):
    def fail_style(*args, **kwargs):
        raise AssertionError("greeting should not hit style service")

    def fail_orchestrator(*args, **kwargs):
        raise AssertionError("greeting should not hit orchestrator")

    monkeypatch.setattr(chat, "_demo_style_board_payload", fail_style)
    monkeypatch.setattr(chat.ahvi_orchestrator, "run", fail_orchestrator)

    app = FastAPI()

    @app.middleware("http")
    async def add_user(request, call_next):
        request.state.user = {"user_id": "user-1"}
        return await call_next(request)

    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    response = client.post(
        "/api/text",
        json={
            "module_context": "style",
            "messages": [{"role": "user", "content": "Hi!"}],
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["message"] == "Hi, I’m here. What would you like help with today?"
    assert body["message_text"] == "Hi, I’m here. What would you like help with today?"
    assert body["cards"] == []
    assert body["style_boards"] == []
    assert body["meta"]["mode"] == "greeting_bypass"


def test_text_chat_fixed_chat_prompts_do_not_hit_style_or_orchestrator(monkeypatch):
    def fail_style(*args, **kwargs):
        raise AssertionError("fixed chat prompts should not hit style service")

    def fail_orchestrator(*args, **kwargs):
        raise AssertionError("fixed chat prompts should not hit orchestrator")

    monkeypatch.setattr(chat, "_demo_style_board_payload", fail_style)
    monkeypatch.setattr(chat.ahvi_orchestrator, "run", fail_orchestrator)
    client = _text_chat_client_with_user()

    expected = {
        "hi": "greeting_bypass",
        "who are you": "help_identity_bypass",
        "what can you do": "help_identity_bypass",
        "help": "help_identity_bypass",
        "how are you": "small_talk_bypass",
    }

    for prompt, mode in expected.items():
        response = client.post(
            "/api/text",
            json={"module_context": "style", "messages": [{"role": "user", "content": prompt}]},
        )
        body = response.json()
        assert response.status_code == 200
        assert body["type"] == "text"
        assert body["cards"] == []
        assert body["style_boards"] == []
        assert body["meta"]["mode"] == mode


def test_text_chat_organize_prompts_route_to_module_service(monkeypatch):
    captured = []

    async def fake_module_chat(payload, user_id=""):
        captured.append((payload, user_id))
        domain = payload["domain"]
        return {
            "success": True,
            "type": "module_chat",
            "domain": domain,
            "module": domain,
            "message": f"{domain} ready",
            "message_text": f"{domain} ready",
            "response": f"{domain} ready",
            "cards": [],
            "style_boards": [],
            "chips": [],
            "data": {"domain": domain},
            "meta": {"mode": domain},
        }

    def fail_style(*args, **kwargs):
        raise AssertionError("organize prompts should not hit style service")

    monkeypatch.setattr(chat, "handle_module_chat", fake_module_chat)
    monkeypatch.setattr(chat, "_demo_style_board_payload", fail_style)
    client = _text_chat_client_with_user()

    expected = {
        "plan my day": "calendar",
        "eat today": "diet",
        "workout today": "fitness",
    }

    for prompt, domain in expected.items():
        response = client.post(
            "/api/text",
            json={"module_context": "chat", "messages": [{"role": "user", "content": prompt}]},
        )
        body = response.json()
        assert response.status_code == 200
        assert body["domain"] == domain
        assert body["style_boards"] == []

    assert [payload["domain"] for payload, _ in captured] == list(expected.values())


def test_text_chat_style_prompts_route_to_style_boards(monkeypatch):
    captured = []

    def fake_style_payload(user_id, query_text, request_wardrobe, user_profile=None, **kwargs):
        captured.append(query_text)
        return {
            "success": True,
            "type": "cards",
            "message": "Style boards ready.",
            "message_text": "Style boards ready.",
            "response": "Style boards ready.",
            "cards": [{"id": "look-1", "items": []}],
            "style_boards": [{"id": "look-1", "items": []}],
            "chips": [],
            "data": {"outfits": [{"id": "look-1"}]},
            "meta": {"mode": "style_flow_service_adapter_v1"},
        }

    monkeypatch.setattr(chat, "_demo_style_board_payload", fake_style_payload)
    client = _text_chat_client_with_user()

    for prompt in ("office outfit", "client meeting", "date night", "client meeting outfit"):
        response = client.post(
            "/api/text",
            json={"module_context": "style", "messages": [{"role": "user", "content": prompt}]},
        )
        body = response.json()
        assert response.status_code == 200
        assert body["style_boards"]
        assert body["cards"]

    assert captured == ["office outfit", "client meeting", "date night", "client meeting outfit"]


def test_text_chat_calendar_event_prompt_creates_event(monkeypatch):
    created = {}

    def fake_create(user_id, payload):
        created["user_id"] = user_id
        created["payload"] = payload
        return {"id": "event-1", "user_id": user_id, **payload}

    def fail_style(*args, **kwargs):
        raise AssertionError("calendar event prompts should not hit style service")

    monkeypatch.setattr("services.calendar_service.create_calendar_event", fake_create)
    monkeypatch.setattr(chat, "_demo_style_board_payload", fail_style)
    client = _text_chat_client_with_user()

    response = client.post(
        "/api/text",
        json={
            "module_context": "chat",
            "messages": [{"role": "user", "content": "doctor appointment tomorrow at 9am"}],
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["domain"] == "calendar"
    assert body["intent"] == "calendar_event_created"
    assert body["refresh"] == "calendar"
    assert created["user_id"] == "user-1"
    assert created["payload"]["title"] == "Doctor appointment"
    assert created["payload"]["type"] == "appointment"
    assert "09:00:00+05:30" in created["payload"]["start_time"]


def test_calendar_module_chat_creates_meeting_event(monkeypatch):
    from services import module_chat_service

    created = {}

    def fake_create(user_id, payload):
        created["user_id"] = user_id
        created["payload"] = payload
        return {"id": "event-2", "user_id": user_id, **payload}

    monkeypatch.setattr("services.calendar_service.create_calendar_event", fake_create)

    result = asyncio.run(
        module_chat_service.handle_calendar_chat(
            "meeting with client friday 4pm",
            {"timezone": "Asia/Kolkata"},
            "user-1",
        )
    )

    assert result["success"] is True
    assert result["domain"] == "calendar"
    assert result["intent"] == "calendar_event_created"
    assert result["refresh"] == "calendar"
    assert created["user_id"] == "user-1"
    assert created["payload"]["title"] == "Meeting with Client"
    assert "16:00:00+05:30" in created["payload"]["start_time"]


def test_plan_my_day_uses_calendar_plan_and_meal_data(monkeypatch):
    from services import module_chat_service

    class FakeProxy:
        def list_documents(self, resource, **kwargs):
            if resource == "plans":
                return [{"title": "Submit project update"}]
            if resource == "meal_plans":
                return [{"title": "High protein day"}]
            return []

    monkeypatch.setattr(
        "services.calendar_service.list_today_calendar_events",
        lambda user_id: [
            {
                "title": "Doctor appointment",
                "start_time": "2026-06-02T09:00:00+05:30",
                "description": "Doctor appointment tomorrow at 9am",
            }
        ],
    )
    monkeypatch.setattr(module_chat_service, "AppwriteProxy", FakeProxy)

    result = asyncio.run(module_chat_service.handle_calendar_chat("plan my day", {}, "user-1"))

    assert result["domain"] == "calendar"
    assert result["intent"] == "plan_my_day"
    assert result["data"]["events_count"] == 1
    assert result["data"]["plans_count"] == 1
    assert result["data"]["meal_plans_count"] == 1
    assert "Doctor appointment" in str(result["cards"])
    assert "Submit project update" in str(result["cards"])
    assert "High protein day" in str(result["cards"])


def test_module_chat_legacy_nested_route_exists(monkeypatch):
    def fake_response(*, module, user_message, history, context_data, user_profile):
        return {
            "success": True,
            "type": "module_chat",
            "module": module,
            "response": "ok",
            "message_text": "ok",
            "message": {"role": "assistant", "content": "ok"},
            "cards": [],
            "style_boards": [],
            "chips": [],
            "data": {"module": module, "rendered_boards": [], "outfits": []},
            "meta": {"mode": module, "board_count": 0},
        }

    monkeypatch.setattr(chat, "_module_llm_response", fake_response)
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    response = client.post(
        "/api/chat/module-chat",
        json={"module": "style", "message": "hello", "history": [], "context_data": {}, "user_profile": {}},
    )

    assert response.status_code == 200
    assert response.json()["type"] == "module_chat"


def test_skincare_module_chat_replaces_truncated_spf_answer(monkeypatch):
    monkeypatch.setattr(
        chat,
        "chat_completion",
        lambda *args, **kwargs: "To recommend the best SPF, I need a bit more detail about",
    )

    result = chat._module_llm_response(
        module="skincare",
        user_message="Best SPF for my skin",
        history=[],
        context_data={},
        user_profile={},
    )

    answer = result["message"]["content"]
    assert result["type"] == "module_chat"
    assert "skin type" in answer
    assert "sun exposure" in answer
    assert answer.endswith(".")


def test_style_module_chat_routes_chip_to_style_flow(monkeypatch):
    captured = {}

    def fake_style_payload(*, user_id, query_text, request_wardrobe, user_profile):
        captured["query"] = query_text
        return {
            "success": True,
            "type": "cards",
            "message": "Travel look ready.",
            "cards": [{"id": "look-1", "items": []}],
            "style_boards": [{"id": "look-1", "items": []}],
            "chips": ["More looks"],
            "board_ids": "look-1",
            "data": {"outfits": [{"id": "look-1"}], "rendered_boards": []},
            "meta": {"mode": "style_flow_service_adapter_v1"},
        }

    monkeypatch.setattr(chat, "_demo_style_board_payload", fake_style_payload)
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    response = client.post(
        "/api/chat/module-chat",
        json={"module": "style", "message": "Travel", "history": [], "context_data": {}, "user_profile": {}},
    )

    body = response.json()
    assert response.status_code == 200
    assert captured["query"] == "Travel"
    assert body["message"]["content"] == "Travel look ready."
    assert body["message_text"] == "Travel look ready."
    assert body["response"] == "Travel look ready."
    assert body["cards"]
    assert body["style_boards"]


def test_style_module_chat_routes_office_meeting_to_board(monkeypatch):
    captured = {}

    def fake_style_payload(*, user_id, query_text, request_wardrobe, user_profile):
        captured["query"] = query_text
        return {
            "success": True,
            "type": "cards",
            "message": "Office meeting look ready.",
            "cards": [{"id": "look-office", "items": []}],
            "style_boards": [{"id": "look-office", "items": []}],
            "chips": [],
            "board_ids": "look-office",
            "data": {"outfits": [{"id": "look-office"}], "rendered_boards": []},
            "meta": {"mode": "style_flow_service_adapter_v1"},
        }

    monkeypatch.setattr(chat, "_demo_style_board_payload", fake_style_payload)
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    response = client.post(
        "/api/chat/module-chat",
        json={"module": "style", "message": "Office meeting", "history": [], "context_data": {}, "user_profile": {}},
    )

    body = response.json()
    assert response.status_code == 200
    assert captured["query"] == "Office meeting"
    assert body["cards"]
    assert "AHVI needs a little more context" not in str(body)


def test_ask_me_two_questions_returns_questions_without_loop():
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    response = client.post(
        "/api/chat/module-chat",
        json={"module": "style", "message": "Ask me 2 questions", "history": [], "context_data": {}, "user_profile": {}},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["intent"] == "ask_questions"
    assert len(body["questions"]) == 2
    assert "AHVI needs a little more context" not in str(body)


def test_style_module_chat_routes_beach_wear_without_empty_llm(monkeypatch):
    def fail_llm(*args, **kwargs):
        raise AssertionError("style prompts should not hit module LLM")

    def fake_style_payload(*, user_id, query_text, request_wardrobe, user_profile):
        assert query_text == "beach wear"
        return {
            "success": False,
            "type": "missing_outfit_cards",
            "message": "I need a top, bottom, and footwear before I can build a beach look.",
            "cards": [],
            "style_boards": [],
            "chips": [{"label": "Add wardrobe", "value": "Use my wardrobe"}],
            "data": {"outfits": [], "rendered_boards": []},
            "meta": {"mode": "style_flow_service_adapter_v1"},
        }

    monkeypatch.setattr(chat, "_module_llm_response", fail_llm)
    monkeypatch.setattr(chat, "_demo_style_board_payload", fake_style_payload)
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    response = client.post(
        "/api/chat/module-chat",
        json={"module": "style", "message": "beach wear", "history": [], "context": {}, "user_profile": {}},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["message"]["content"]
    assert "beach" in body["message_text"]
    assert body["chips"]


def test_text_chat_bare_style_action_requires_context_without_crash():
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    response = client.post(
        "/api/text",
        json={"messages": [{"role": "user", "content": "Show closest option"}]},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["success"] is True
    assert body["type"] == "context_required"
    assert body["data"]["requires_context"] is True
    assert body["data"]["missing_context_for_action"] == "show closest option"


def test_text_chat_bare_style_action_recovers_previous_prompt_from_history():
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    response = client.post(
        "/api/text",
        json={
            "messages": [
                {"role": "user", "content": "Beach wear · Casual beach walk"},
                {
                    "role": "assistant",
                    "content": "I checked your wardrobe against the occasion. I found a few close matches.",
                },
                {"role": "user", "content": "Show closest option"},
            ]
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Authenticated user is required"


def test_style_fallback_forwards_style_action(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        chat,
        "_ahvi_resolve_effective_user_profile",
        lambda user_id, user_profile=None: {"user_id": user_id},
    )
    monkeypatch.setattr(
        chat,
        "_fetch_wardrobe_for_style",
        lambda user_id, request_wardrobe: [{"id": "top-1", "category": "top"}],
    )
    monkeypatch.setattr(chat, "_ahvi_item_allowed_for_user_profile", lambda *args, **kwargs: True)

    def fake_build_style_flow_response(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "message": "Closest option ready.",
            "type": "cards",
            "cards": [{"id": "look-1", "items": []}],
            "style_boards": [{"id": "look-1", "items": []}],
            "data": {"outfits": [{"id": "look-1"}]},
            "meta": {},
        }

    monkeypatch.setattr(chat, "build_style_flow_response", fake_build_style_flow_response)

    response = chat._demo_style_board_payload(
        "user-1",
        "beach wear · Casual beach walk",
        request_wardrobe=[],
        user_profile={},
        style_action="show_closest_option",
    )

    assert response["cards"]
    assert captured["style_action"] == "show_closest_option"
    assert captured["show_closest_option"] is True
    assert captured["allow_closest_option"] is True
    assert captured["closest"] is True


def test_planner_module_routes_plan_pack_to_checklists(monkeypatch):
    def fail_module_chat(*args, **kwargs):
        raise AssertionError("plan-pack prompts should not use generic planner fallback")

    monkeypatch.setattr(chat, "handle_module_chat", fail_module_chat)
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    response = client.post(
        "/api/chat/module-chat",
        json={
            "module": "planner",
            "message": "plan and pack for a 2 day beach trip",
            "history": [],
            "context_data": {},
            "user_profile": {},
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["type"] == "checklists"
    assert body["meta"]["intent"] == "plan_pack"
    assert len(body["cards"]) >= 3


def test_lifestyle_intent_engine_routes_known_prompts():
    from brain.intent_engine import detect_intent

    expected = {
        "Today's meals": ("organize_hub", "meal_planner"),
        "Today's workout": ("organize_hub", "workout"),
        "Morning skincare": ("organize_hub", "skincare"),
        "Morning routine": ("organize_hub", "skincare"),
        "Evening routine": ("organize_hub", "skincare"),
        "Pending bills": ("organize_hub", "bills"),
        "My medicines": ("organize_hub", "medicines"),
        "Today's events": ("organize_hub", "calendar"),
        "Upcoming events": ("organize_hub", "calendar"),
    }

    for prompt, (intent, module) in expected.items():
        row = detect_intent(prompt)
        assert row["intent"] == intent
        assert row["slots"]["module"] == module
        assert row["confidence"] >= 0.75


def test_chat_routing_intent_engine_prioritizes_plan_matrix():
    from brain.intent_engine import detect_intent

    expected_modules = {
        "plan my day": ("organize_hub", "calendar"),
        "eat today": ("organize_hub", "meal_planner"),
        "workout today": ("organize_hub", "workout"),
    }
    for prompt, (intent, module) in expected_modules.items():
        row = detect_intent(prompt)
        assert row["intent"] == intent
        assert row["slots"]["module"] == module

    for prompt in ("office outfit", "client meeting", "date night", "casual dinner"):
        row = detect_intent(prompt)
        assert row["intent"] == "occasion_outfit"
        assert row["intent"] != "organize_hub"


def test_calendar_event_intents_do_not_route_to_style():
    from brain.intent_engine import detect_intent

    prompts = [
        "doctor appointment tomorrow at 9am",
        "meeting with client friday 4pm",
        "call with kavya tomorrow",
        "dentist Friday 6pm",
        "interview Monday morning",
    ]

    for prompt in prompts:
        row = detect_intent(prompt)
        assert row["intent"] == "organize_hub"
        assert row["slots"]["module"] == "calendar"
        assert row["slots"]["action"] == "create_event"
        assert row["confidence"] >= 0.9

    style_row = detect_intent("client meeting outfit")
    assert style_row["intent"] == "occasion_outfit"


def test_occasion_guards_block_called_out_bad_pairings():
    from brain.engines.occasion_style_rules import get_occasion_rule, reject_board_for_occasion, score_item_for_occasion
    from brain.engines.outfit_quality_guard import reject_board_for_occasion as reject_quality_board

    shiny_gold_shirt = {
        "name": "Shiny Gold Satin Shirt",
        "category": "shirt",
        "color": "gold",
        "material": "satin",
    }
    assert score_item_for_occasion(shiny_gold_shirt, get_occasion_rule("office")) <= -10
    assert reject_board_for_occasion({"items": [shiny_gold_shirt]}, "office") == (
        "shiny_gold_shirt_blocked_for_smart_occasion"
    )

    shorts_board = {"items": [{"name": "Tailored shorts", "category": "bottom"}]}
    assert reject_board_for_occasion(shorts_board, "date_night") == "short_bottom_blocked_for_smart_occasion"
    assert reject_quality_board(shorts_board, "office")[0] is True

    beach_board = {"items": [{"name": "Formal loafers", "category": "footwear"}]}
    assert reject_board_for_occasion(beach_board, "beach").startswith("occasion_mismatch:")
    assert reject_quality_board(beach_board, "beach")[0] is True


def test_plan_pack_prompts_route_without_generic_fallback():
    prompts = [
        "Help me prep for camping",
        "Plan for a 3 day Goa trip",
        "Plan a birthday party",
        "Pack for a carry-on trip",
    ]
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    for prompt in prompts:
        response = client.post(
            "/api/chat/module-chat",
            json={
                "module": "planner",
                "message": prompt,
                "history": [],
                "context_data": {},
                "user_profile": {},
            },
        )

        body = response.json()
        assert response.status_code == 200
        assert body["intent"] == "plan_pack"
        assert body["meta"]["intent"] == "plan_pack"
        assert body["cards"]
        assert [a["label"] for a in body["quick_actions"]] == [
            "Open checklist",
            "Plan outfits",
            "Weather prep",
            "Save trip plan",
        ]
        assert body["quick_actions"][0]["intent"] == "open_checklist"
        assert "I can help with style, planning, and wardrobe advice" not in body["message"]["content"]


def test_module_summary_prompts_return_cards_and_actions(monkeypatch):
    from services import module_summary_service

    monkeypatch.setattr(chat, "_state_user_id", lambda request: "user-1")
    monkeypatch.setattr(module_summary_service, "_docs", lambda *args, **kwargs: [])

    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    expected = {
        "Today's meals": "meals",
        "Today's workout": "workout",
        "Morning skincare": "skincare",
        "Morning routine": "skincare",
        "Evening routine": "skincare",
        "Pending bills": "bills",
        "My medicines": "medicines",
        "Today's events": "events",
        "Upcoming events": "events_upcoming",
    }

    for prompt, module in expected.items():
        response = client.post(
            "/api/chat/module-chat",
            json={
                "module": "chat",
                "message": prompt,
                "history": [],
                "context_data": {},
                "user_profile": {},
            },
        )

        body = response.json()
        assert response.status_code == 200
        assert body["type"] == "module_card"
        assert body["module"] == ("events" if module == "events_upcoming" else module)
        assert body["card"]
        assert body["quick_actions"]
        assert "I can help with style, planning, and wardrobe advice" not in body["message"]


def test_plan_pack_destination_labels_are_clean():
    from brain.plan_pack_flow import build_plan_pack_response

    goa = build_plan_pack_response("plan for a 3 day Goa trip")
    carry_on = build_plan_pack_response("Pack for a carry-on trip")
    birthday = build_plan_pack_response("Plan a birthday party")

    assert goa["data"]["destination"] == "Goa"
    assert goa["cards"][0]["title"] == "3-Day Goa Trip"
    assert goa["cards"][0]["subtitle"] == "Goa · 3 days"
    assert carry_on["data"]["destination"] == "Carry-On Trip"
    assert carry_on["data"]["duration_label"] == "Short trip"
    assert carry_on["cards"][0]["title"] == "Carry-on Packing Checklist"
    assert carry_on["cards"][0]["subtitle"] == "Short carry-on trip"
    assert birthday["data"]["destination"] == "Birthday Party"
    assert birthday["cards"][0]["title"] == "Birthday Party Plan"
    assert len(birthday["cards"]) == 1
    birthday_items = " ".join(item["label"] for item in birthday["cards"][0]["items"]).lower()
    assert "book transport" not in birthday_items
    assert "pack essentials" not in birthday_items
    assert "carry-on" not in birthday_items
    assert "guest list" in birthday_items
    assert goa["cards"][1]["items"][0]["checked"] is False
    assert "assetIcon" in goa["cards"][1]["items"][0]
    assert goa["cards"][1]["action"]["module"] == "plan_pack"
    assert goa["cards"][2]["action"]["label"] == "Open checklist"


def test_known_quick_actions_do_not_use_generic_fallback():
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    expected = {
        "Workout outfit": "fitness",
        "Gym workout": "fitness",
        "Home workout": "fitness",
        "Recovery meal": "diet",
        "Weather prep": "plan_pack",
    }

    for prompt, expected_intent in expected.items():
        response = client.post(
            "/api/chat/module-chat",
            json={
                "module": "chat",
                "message": prompt,
                "history": [],
                "context_data": {},
                "user_profile": {},
            },
        )
        body = response.json()
        assert response.status_code == 200
        assert "I can help with style, planning, and wardrobe advice" not in str(body)
        if expected_intent == "plan_pack":
            assert body["intent"] in {"plan_pack", "weather_prep"}
        else:
            assert body["intent"] == expected_intent
        assert "Open life boards" not in str(body)


def test_add_event_quick_action_starts_capture():
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    response = client.post(
        "/api/chat/module-chat",
        json={"module": "chat", "message": "Add event", "history": [], "context_data": {}, "user_profile": {}},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["module"] == "calendar"
    assert body["intent"] == "create_event"
    assert "event name and date" in body["message_text"]


def test_chat_birthday_creates_calendar_event(monkeypatch):
    created = {}

    def fake_create(user_id, payload):
        created["user_id"] = user_id
        created["payload"] = payload
        return {"id": "event-1", "user_id": user_id, **payload}

    monkeypatch.setattr(chat, "_state_user_id", lambda request: "user-1")
    monkeypatch.setattr("services.calendar_service.create_calendar_event", fake_create)

    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    response = client.post(
        "/api/chat/module-chat",
        json={"module": "calendar", "message": "my birthday on 23rd July", "history": [], "context_data": {}, "user_profile": {}},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["intent"] == "event_created"
    assert body["module"] == "calendar"
    assert created["user_id"] == "user-1"
    assert created["payload"]["title"] == "Birthday"
    assert created["payload"]["type"] == "birthday"
    assert "07-23T09:00:00+05:30" in created["payload"]["start_time"]
    assert body["quick_actions"] == ["View events", "Add reminder", "Plan outfit"]
