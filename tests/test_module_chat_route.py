import asyncio
import json

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


def _fake_style_reasoning_json(prompt, **kwargs):
    mode = "visual_inspiration" if "visual inspiration" in prompt.lower() else "style_advice"
    return json.dumps(
        {
            "mode": mode,
            "occasion": "coffee date",
            "goal": "Look relaxed and considered.",
            "atmosphere": "easy and warm",
            "emotion_state": "social",
            "stylist_advice": "Keep it easy, clean, and approachable with one polished detail.",
            "what_to_avoid": ["anything too stiff", "loud logos"],
            "visual_directions": [
                {
                    "title": "Relaxed Oxford",
                    "description": "Oxford shirt with dark denim and clean sneakers.",
                    "palette": ["navy", "white", "tan"],
                    "pieces": ["Oxford shirt", "dark denim", "clean sneakers"],
                    "style_note": "Tidy without feeling formal.",
                },
                {
                    "title": "Knit Polo Polish",
                    "description": "Soft knit top with straight trousers and loafers.",
                    "palette": ["cream", "olive", "brown"],
                    "pieces": ["knit polo", "straight trousers", "loafers"],
                    "style_note": "Texture keeps it warm.",
                },
                {
                    "title": "Soft Layered Casual",
                    "description": "Light jacket over a simple base.",
                    "palette": ["stone", "blue", "charcoal"],
                    "pieces": ["light jacket", "plain tee", "chinos"],
                    "style_note": "Useful if the setting shifts.",
                },
            ],
            "follow_up_question": None,
            "confidence": 0.91,
        }
    )


def _fake_style_pairing_json(prompt, **kwargs):
    return json.dumps(
        {
            "mode": "style_pairing",
            "anchor_item": {"name": "white shirt", "category": "shirt", "color": "white"},
            "stylist_reasoning": "A white shirt is strongest when you decide the mood first: crisp, relaxed, evening, or summer. Keep the shirt as the clean anchor and change the base, shoe, and texture around it.",
            "pairing_routes": [
                {
                    "title": "Smart Casual",
                    "use_case": "office-adjacent days",
                    "strategy": "Relax the shirt with chinos and clean shoes.",
                    "items": ["white shirt", "tan chinos", "brown loafers"],
                    "palette": ["white", "tan", "brown"],
                    "why_it_works": "The chinos soften the shirt while loafers keep it intentional.",
                    "avoid": ["shiny ties"],
                    "styling_tip": "Leave the collar open.",
                },
                {
                    "title": "Business Casual",
                    "use_case": "meetings",
                    "strategy": "Add structure around the shirt.",
                    "items": ["white shirt", "grey trousers", "black loafers"],
                    "palette": ["white", "grey", "black"],
                    "why_it_works": "The tailored base makes the shirt credible.",
                    "avoid": ["distressed denim"],
                    "styling_tip": "Tuck it cleanly.",
                },
                {
                    "title": "Weekend Clean",
                    "use_case": "coffee or errands",
                    "strategy": "Use denim and sneakers.",
                    "items": ["white shirt", "dark denim", "clean sneakers"],
                    "palette": ["white", "blue", "stone"],
                    "why_it_works": "Denim keeps the shirt approachable.",
                    "avoid": ["overly formal shoes"],
                    "styling_tip": "Roll sleeves once.",
                },
                {
                    "title": "Evening Minimal",
                    "use_case": "dinner",
                    "strategy": "Use contrast and quiet accessories.",
                    "items": ["white shirt", "black trousers", "sleek shoes"],
                    "palette": ["white", "black", "charcoal"],
                    "why_it_works": "The contrast makes the shirt look deliberate.",
                    "avoid": ["loud belts"],
                    "styling_tip": "Keep accessories minimal.",
                },
                {
                    "title": "Summer Relaxed",
                    "use_case": "warm days",
                    "strategy": "Use breathable textures.",
                    "items": ["white shirt", "linen trousers", "canvas sneakers"],
                    "palette": ["white", "ecru", "olive"],
                    "why_it_works": "Light fabric makes the shirt feel easy.",
                    "avoid": ["heavy formal trousers"],
                    "styling_tip": "Wear it slightly open over a tee.",
                },
            ],
            "what_to_avoid": ["five tiny color variations", "over-formal styling only"],
            "next_actions": ["Use my wardrobe", "Show visual inspiration", "Find missing pieces"],
            "follow_up_question": None,
            "confidence": 0.92,
        }
    )


def _fake_visual_first_reasoning(*, query, intent, context, **kwargs):
    mode = str(intent.get("intent") if isinstance(intent, dict) else intent)
    assert mode == "visual_inspiration"
    transition = context.get("multi_event") if isinstance(context, dict) else None
    directions = [
        {
            "title": f"Direction {index}",
            "archetype": f"Archetype {index}",
            "description": "A complete image-backed style direction.",
            "pieces": ["shirt", "trousers", "shoes"],
            "image_url": f"https://example.com/look-{index}.jpg",
            "asset_id": f"asset-{index}",
            "complete_the_look": [],
        }
        for index in range(1, 4)
    ]
    return {
        "mode": "visual_inspiration",
        "occasion": context.get("occasion") if isinstance(context, dict) else "",
        "advice": "Three visual directions to start from.",
        "visual_directions": directions,
        "visual_inspiration_board": {"title": "Visual inspiration", "directions": directions},
        "transition_plan": transition if isinstance(transition, dict) else None,
        "is_transition": bool(transition),
        "cta": [
            {
                "label": "Use my wardrobe",
                "value": f"Use my wardrobe for: {query}",
            }
        ],
        "should_generate_board": False,
        "should_use_wardrobe": False,
    }


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


def test_text_chat_style_prompts_route_to_advice_first(monkeypatch):
    def fail_style(*args, **kwargs):
        raise AssertionError("general style prompts should not hit style service")

    monkeypatch.setattr("services.style_reasoning_engine.generate_text", _fake_style_reasoning_json)
    monkeypatch.setattr(chat, "_demo_style_board_payload", fail_style)
    client = _text_chat_client_with_user()

    for prompt in ("office outfit", "client meeting", "date night", "client meeting outfit"):
        response = client.post(
            "/api/text",
            json={"module_context": "style", "messages": [{"role": "user", "content": prompt}]},
        )
        body = response.json()
        assert response.status_code == 200
        assert body["type"] == "stylist_advice"
        assert body["style_boards"] == []
        assert body["cards"]
        assert len(body["data"]["visual_directions"]) == 3
        assert body["meta"]["style_mode"] == "style_advice"


def test_text_chat_style_pairing_returns_pairing_routes(monkeypatch):
    def fail_style(*args, **kwargs):
        raise AssertionError("style pairing should not hit wardrobe board service")

    monkeypatch.setattr("services.style_reasoning_engine.generate_text", _fake_style_pairing_json)
    monkeypatch.setattr(chat, "_demo_style_board_payload", fail_style)
    client = _text_chat_client_with_user()

    response = client.post(
        "/api/text",
        json={"module_context": "style", "messages": [{"role": "user", "content": "What to pair with a white shirt?"}]},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["type"] == "stylist_advice"
    assert body["intent"] == "style_pairing"
    assert body["style_boards"] == []
    assert body["data"]["anchor_item"]["name"] == "white shirt"
    assert len(body["data"]["pairing_routes"]) >= 4
    assert len(body["data"]["visual_directions"]) == len(body["data"]["pairing_routes"])
    assert all(card["type"] == "visual_direction" for card in body["cards"][1:])
    assert body["meta"]["style_mode"] == "style_pairing"


def test_text_chat_general_style_advice_bypasses_wardrobe_style(monkeypatch):
    def fail_style(*args, **kwargs):
        raise AssertionError("general style advice should not hit style service")

    def fail_orchestrator(*args, **kwargs):
        raise AssertionError("general style advice should not hit orchestrator")

    monkeypatch.setattr("services.style_reasoning_engine.generate_text", _fake_style_reasoning_json)
    monkeypatch.setattr(chat, "_demo_style_board_payload", fail_style)
    monkeypatch.setattr(chat.ahvi_orchestrator, "run", fail_orchestrator)
    client = _text_chat_client_with_user()

    prompts = [
        "What should I wear to a coffee date?",
        "What should I wear to a Christian funeral?",
        "What colors suit warm skin?",
        "I have a pear body type.",
        "Recommend shoes for this outfit.",
    ]

    for prompt in prompts:
        response = client.post(
            "/api/text",
            json={"module_context": "style", "messages": [{"role": "user", "content": prompt}]},
        )
        body = response.json()
        assert response.status_code == 200
        assert body["type"] == "stylist_advice"
        assert body["style_boards"] == []
        assert len(body["data"]["visual_directions"]) == 3
        assert body["meta"]["mode"] == "style_reasoning"
        assert body["meta"]["style_mode"] in {
            "style_advice",
            "color_body_advice",
            "color_advice",
            "body_proportion_advice",
            "shopping_assist",
        }
        assert body["meta"]["wardrobe_lookup"] is False
        assert "Use my wardrobe" in [chip["label"] for chip in body["chips"]]


def test_text_chat_visual_inspiration_returns_direction_cards(monkeypatch):
    def fail_style(*args, **kwargs):
        raise AssertionError("visual inspiration should not hit wardrobe board service")

    monkeypatch.setattr("services.style_reasoning_engine.generate_text", _fake_style_reasoning_json)
    monkeypatch.setattr(chat, "_demo_style_board_payload", fail_style)
    client = _text_chat_client_with_user()

    response = client.post(
        "/api/text",
        json={
            "module_context": "style",
            "messages": [{"role": "user", "content": "Show visual inspiration for coffee date"}],
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["type"] == "stylist_advice"
    assert body["style_boards"] == []
    assert body["meta"]["style_mode"] == "visual_inspiration"
    assert len(body["data"]["visual_directions"]) == 3
    assert all(direction.get("archetype") for direction in body["data"]["visual_directions"])
    assert len([card for card in body["cards"] if card["type"] == "visual_direction"]) == 3
    assert all(card.get("archetype") for card in body["cards"] if card["type"] == "visual_direction")


def test_visual_first_toggle_routes_open_style_prompts_to_image_cards(monkeypatch):
    monkeypatch.setenv("STYLE_DEFAULT_VISUAL_INSPIRATION", "true")
    monkeypatch.setattr(chat.style_reasoning_engine, "reason", _fake_visual_first_reasoning)
    monkeypatch.setattr(
        chat,
        "_demo_style_board_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("visual-first prompts must not hit wardrobe generation")
        ),
    )
    client = _text_chat_client_with_user()

    prompts = (
        "cousin wedding",
        "conference talk and cocktails",
        "client meeting tomorrow",
        "coffee date",
        "gym then brunch",
        "airport outfit",
    )
    for prompt in prompts:
        response = client.post(
            "/api/text",
            json={
                "module_context": "style",
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        body = response.json()
        direction_cards = [
            card for card in body.get("cards", []) if card.get("type") == "visual_direction"
        ]

        assert response.status_code == 200, body
        assert body["intent"] == "visual_inspiration"
        assert body["meta"]["style_mode"] == "visual_inspiration"
        assert len(body["data"]["visual_directions"]) == 3
        assert len(direction_cards) == 3
        assert all(card.get("image_url") and card.get("asset_id") for card in direction_cards)
        if prompt in {"conference talk and cocktails", "gym then brunch"}:
            assert body["data"]["is_transition"] is True
            assert body["data"]["transition_plan"]


def test_visual_first_toggle_applies_to_module_chat(monkeypatch):
    monkeypatch.setenv("STYLE_DEFAULT_VISUAL_INSPIRATION", "true")
    monkeypatch.setattr(chat.style_reasoning_engine, "reason", _fake_visual_first_reasoning)
    monkeypatch.setattr(
        chat,
        "_demo_style_board_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("module style prompt must use visual inspiration")
        ),
    )
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    response = client.post(
        "/api/chat/module-chat",
        json={
            "module": "style",
            "message": "cousin wedding",
            "history": [],
            "context_data": {},
            "user_profile": {},
        },
    )
    body = response.json()

    assert response.status_code == 200, body
    assert body["intent"] == "visual_inspiration"
    assert len([card for card in body["cards"] if card["type"] == "visual_direction"]) == 3


def test_visual_first_toggle_preserves_explicit_wardrobe_actions(monkeypatch):
    monkeypatch.setenv("STYLE_DEFAULT_VISUAL_INSPIRATION", "true")
    captured = []

    def fake_style_payload(user_id, query_text, request_wardrobe, user_profile=None, **kwargs):
        captured.append((query_text, kwargs.get("style_action")))
        return {
            "success": True,
            "type": "cards",
            "message": "Wardrobe style board ready.",
            "cards": [{"id": "look-1", "items": []}],
            "style_boards": [{"id": "look-1", "items": []}],
            "chips": [],
            "data": {"outfits": [{"id": "look-1"}]},
            "meta": {"mode": "style_flow_service_adapter_v1"},
        }

    monkeypatch.setattr(chat, "_demo_style_board_payload", fake_style_payload)
    monkeypatch.setattr(
        chat.style_reasoning_engine,
        "reason",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("explicit wardrobe actions must skip inspiration reasoning")
        ),
    )
    client = _text_chat_client_with_user()

    for prompt in (
        "Use my wardrobe for: coffee date",
        "Show wardrobe matches for: coffee date",
        "Build from my wardrobe for: coffee date",
    ):
        response = client.post(
            "/api/text",
            json={
                "module_context": "style",
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        body = response.json()
        assert response.status_code == 200, body
        assert body["data"]["intent"] == "wardrobe_style"
        assert body["meta"]["forced_pipeline"] == "outfit_pipeline"

    assert [action for _, action in captured] == ["use_wardrobe"] * 3


def test_visual_first_toggle_excludes_non_generation_style_modes(monkeypatch):
    monkeypatch.setenv("STYLE_DEFAULT_VISUAL_INSPIRATION", "true")

    excluded = {
        "What colors suit warm skin?": "color_advice",
        "What is smart casual?": "style_education",
        "What to pair with a white shirt?": "style_pairing",
        "Find missing pieces for this look": "shopping_assist",
    }
    for prompt, intent in excluded.items():
        assert (
            chat._should_default_visual_inspiration(
                prompt,
                intent=intent,
                module_context="style",
            )
            is False
        )


def test_text_chat_explicit_wardrobe_style_still_hits_style_service(monkeypatch):
    captured = []

    def fake_style_payload(user_id, query_text, request_wardrobe, user_profile=None, **kwargs):
        captured.append(query_text)
        return {
            "success": True,
            "type": "cards",
            "message": "Wardrobe style board ready.",
            "message_text": "Wardrobe style board ready.",
            "response": "Wardrobe style board ready.",
            "cards": [{"id": "look-1", "items": []}],
            "style_boards": [{"id": "look-1", "items": []}],
            "chips": [],
            "data": {"outfits": [{"id": "look-1"}]},
            "meta": {"mode": "style_flow_service_adapter_v1"},
        }

    monkeypatch.setattr(chat, "_demo_style_board_payload", fake_style_payload)
    client = _text_chat_client_with_user()

    response = client.post(
        "/api/text",
        json={
            "module_context": "style",
            "messages": [{"role": "user", "content": "Use my wardrobe for a coffee date."}],
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["style_boards"]
    assert captured == ["Use my wardrobe for a coffee date."]


def test_text_chat_wardrobe_action_overrides_visual_inspiration(monkeypatch):
    captured = []

    def fake_style_payload(user_id, query_text, request_wardrobe, user_profile=None, **kwargs):
        captured.append((query_text, kwargs))
        return {
            "success": True,
            "type": "cards",
            "message": "Wardrobe style board ready.",
            "message_text": "Wardrobe style board ready.",
            "response": "Wardrobe style board ready.",
            "cards": [{"id": "look-1", "items": []}],
            "style_boards": [{"id": "look-1", "items": []}],
            "chips": [],
            "data": {"outfits": [{"id": "look-1"}]},
            "meta": {"mode": "style_flow_service_adapter_v1"},
        }

    def fail_reasoning(*args, **kwargs):
        raise AssertionError("wardrobe action must skip visual inspiration reasoning")

    monkeypatch.setattr(chat, "_demo_style_board_payload", fake_style_payload)
    monkeypatch.setattr(chat.style_reasoning_engine, "reason", fail_reasoning)
    client = _text_chat_client_with_user()

    response = client.post(
        "/api/text",
        json={
            "module_context": "style",
            "messages": [
                {
                    "role": "user",
                    "content": "Use my wardrobe for: show visual inspiration for coffee date",
                }
            ],
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["style_boards"]
    assert captured[0][0] == "Use my wardrobe for coffee date"
    assert captured[0][1]["style_action"] == "use_wardrobe"
    assert body["data"]["intent"] == "wardrobe_style"
    assert body["meta"]["forced_pipeline"] == "outfit_pipeline"


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


def test_calendar_context_routes_event_shaped_meal_words_to_calendar(monkeypatch):
    created = []

    monkeypatch.setattr("services.calendar_service.find_existing_event", lambda *args, **kwargs: None)

    def fake_create(user_id, payload):
        event = {"id": f"event-{len(created) + 1}", "user_id": user_id, **payload}
        created.append(event)
        return event

    monkeypatch.setattr("services.calendar_service.create_calendar_event", fake_create)
    client = _text_chat_client_with_user()

    prompts = [
        "Dentist 18:00",
        "Dinner 20:00",
        "Lunch with Ravi 13:30",
        "Breakfast meeting tomorrow at 9",
        "Gym 07:00",
        "Dinner with Meghna Friday 8pm",
    ]
    for prompt in prompts:
        response = client.post(
            "/api/chat/module-chat",
            json={
                "domain": "calendar",
                "module": "calendar",
                "message": f"Occasion: Event\n\n{prompt}",
                "history": [],
                "context_data": {},
                "user_profile": {},
            },
        )
        body = response.json()
        assert response.status_code == 200
        assert body["intent"] == "calendar_event_created"
        assert body["domain"] == "calendar"

    assert len(created) == len(prompts)


def test_calendar_consecutive_dentist_and_dinner_create_two_events(monkeypatch):
    created = []

    def fail_diet_handler(**kwargs):
        raise AssertionError("Calendar event turns must not invoke the Diet board handler")

    monkeypatch.setattr("services.calendar_service.find_existing_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat, "_build_visual_board_envelope", fail_diet_handler)

    def fake_create(user_id, payload):
        event = {"id": f"event-{len(created) + 1}", "user_id": user_id, **payload}
        created.append(event)
        return event

    monkeypatch.setattr("services.calendar_service.create_calendar_event", fake_create)
    client = _text_chat_client_with_user()

    history = []
    responses = []
    for prompt in ("Dentist 18:00", "Dinner 20:00"):
        decorated = f"Occasion: Event\n\n{prompt}"
        response = client.post(
            "/api/chat/module-chat",
            json={
                "domain": "calendar",
                "module": "calendar",
                "message": decorated,
                "history": history,
                "context_data": {},
                "user_profile": {},
            },
        )
        body = response.json()
        responses.append(body)
        history.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": body["message_text"]},
            ]
        )

    assert [body["intent"] for body in responses] == [
        "calendar_event_created",
        "calendar_event_created",
    ]
    assert created[0]["title"].startswith("Dentist")
    assert created[1]["title"] == "Dinner"


def test_explicit_diet_prompts_and_outside_calendar_dinner_stay_non_calendar(monkeypatch):
    created = []
    monkeypatch.setattr(
        "services.calendar_service.create_calendar_event",
        lambda user_id, payload: created.append(payload),
    )

    diet_prompts = [
        "What should I eat for dinner?",
        "Plan a high-protein dinner",
        "How many calories should dinner have?",
        "Give me dinner meal ideas",
    ]
    for prompt in diet_prompts:
        assert chat._detect_visual_board_type(prompt, "calendar") == "diet_plan"
        assert chat._looks_like_event_create(prompt) is False

    assert chat._looks_like_event_create("Dinner") is False
    assert chat._detect_visual_board_type("Dinner", "chat") == "diet_plan"
    assert created == []


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

    def fake_style_payload(*, user_id, query_text, request_wardrobe, user_profile, **kwargs):
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


def test_conversational_outfit_phrases_stay_on_module_chat_style_path(monkeypatch):
    def fake_style_payload(*, user_id, query_text, request_wardrobe, user_profile, **kwargs):
        return {
            "success": True,
            "type": "cards",
            "message": "Style look ready.",
            "cards": [{"id": "look-1", "items": []}],
            "style_boards": [{"id": "look-1", "items": []}],
            "chips": [],
            "board_ids": "look-1",
            "data": {"outfits": [{"id": "look-1"}], "rendered_boards": []},
            "meta": {"mode": "style_flow_service_adapter_v1"},
        }

    monkeypatch.setattr(chat, "_demo_style_board_payload", fake_style_payload)
    client = _text_chat_client_with_user()
    prompts = [
        "build me an outfit",
        "build an outfit for tomorrow",
        "put together an outfit for dinner",
        "build an outfit around this shirt",
    ]

    for prompt in prompts:
        response = client.post(
            "/api/module-chat",
            json={
                "module": "style",
                "message": prompt,
                "history": [],
                "context_data": {"anchor_item_id": "shirt-1"},
                "user_profile": {},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert "Try-On is coming soon" not in str(body)
        assert body["message_text"]


def test_explicit_outfit_request_skips_semantic_provider_without_carried_context(monkeypatch):
    semantic_calls = []

    def fail_semantic_provider(**kwargs):
        semantic_calls.append(kwargs)
        raise AssertionError("explicit one-turn outfit request must not call semantic provider")

    monkeypatch.setattr(chat, "resolve_semantic_intent", fail_semantic_provider)
    monkeypatch.setattr(
        chat,
        "_demo_style_board_payload",
        lambda **kwargs: {
            "success": True,
            "type": "cards",
            "message": "Style look ready.",
            "cards": [{"id": "look-1", "items": []}],
            "style_boards": [{"id": "look-1", "items": []}],
            "chips": [],
            "board_ids": "look-1",
            "data": {"outfits": [{"id": "look-1"}], "rendered_boards": []},
            "meta": {"mode": "style_flow_service_adapter_v1"},
        },
    )
    client = _text_chat_client_with_user()
    prompt = "office outfit using my wardrobe"
    response = client.post(
        "/api/module-chat",
        json={
            "module": "style",
            "message": prompt,
            "history": [{"role": "user", "content": prompt}],
            "context_data": {},
            "user_profile": {},
        },
    )

    assert response.status_code == 200
    assert response.json()["cards"]
    assert semantic_calls == []


def test_explicit_outfit_followup_keeps_semantic_provider_for_carried_context(monkeypatch):
    semantic_calls = []

    def semantic_provider(**kwargs):
        semantic_calls.append(kwargs)
        return None

    monkeypatch.setattr(chat, "resolve_semantic_intent", semantic_provider)
    monkeypatch.setattr(
        chat,
        "_demo_style_board_payload",
        lambda **kwargs: {
            "success": True,
            "type": "cards",
            "message": "Style look ready.",
            "cards": [{"id": "look-1", "items": []}],
            "style_boards": [{"id": "look-1", "items": []}],
            "chips": [],
            "board_ids": "look-1",
            "data": {"outfits": [{"id": "look-1"}], "rendered_boards": []},
            "meta": {"mode": "style_flow_service_adapter_v1"},
        },
    )
    client = _text_chat_client_with_user()
    response = client.post(
        "/api/module-chat",
        json={
            "module": "style",
            "message": "build an outfit for dinner",
            "history": [
                {"role": "user", "content": "I have a client meeting tomorrow"},
                {"role": "assistant", "content": "What setting is it in?"},
                {"role": "user", "content": "build an outfit for dinner"},
            ],
            "context_data": {},
            "user_profile": {},
        },
    )

    assert response.status_code == 200
    assert len(semantic_calls) == 1


def test_style_module_chat_routes_office_meeting_to_board(monkeypatch):
    captured = {}

    def fail_semantic_provider(**kwargs):
        raise AssertionError("direct office prompts must not call semantic provider")

    def fake_style_payload(*, user_id, query_text, request_wardrobe, user_profile, **kwargs):
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
    monkeypatch.setattr(chat, "resolve_semantic_intent", fail_semantic_provider)
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

    def fake_style_payload(*, user_id, query_text, request_wardrobe, user_profile, **kwargs):
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
        assert row["intent"] == "style_advice"
        assert row["intent"] != "organize_hub"


def test_stylist_first_intent_engine_routes_advice_before_wardrobe():
    from brain.intent_engine import detect_intent

    expected = {
        "What should I wear to a coffee date?": "style_advice",
        "What should I wear to a Christian funeral?": "style_advice",
        "What should I wear to my cousin's temple lunch after a client pitch?": "style_advice",
        "What colors suit warm skin?": "color_advice",
        "I have a pear body type.": "body_proportion_advice",
        "Use my wardrobe for a coffee date.": "wardrobe_style",
        "Build a look from my wardrobe.": "wardrobe_style",
        "Use my wardrobe for Smart Casual with white shirt.": "wardrobe_style",
        "Recommend shoes for this outfit.": "shopping_assist",
        "What is smart casual?": "style_education",
        "What to pair with a white shirt?": "style_pairing",
        "How do I style black loafers?": "style_pairing",
    }

    for prompt, intent in expected.items():
        row = detect_intent(prompt)
        assert row["intent"] == intent


def test_style_reasoning_engine_schema_and_decisions(monkeypatch):
    from services.style_reasoning_engine import style_reasoning_engine

    monkeypatch.setattr("services.style_reasoning_engine.generate_text", _fake_style_reasoning_json)

    funeral = style_reasoning_engine.reason(
        query="What should I wear to a Christian funeral?",
        intent="style_advice",
    )
    assert funeral["mode"] == "style_advice"
    assert funeral["should_use_wardrobe"] is False
    assert funeral["should_generate_board"] is False
    assert funeral["advice"]
    assert funeral["meta"]["source"] == "style_reasoning_engine"
    assert funeral["meta"]["reason"] in {"sensitive_occasion", "style_advice"}
    assert funeral["meta"]["goal"]
    assert funeral["meta"]["atmosphere"]
    assert len(funeral["visual_directions"]) == 3

    coffee = style_reasoning_engine.reason(
        query="What should I wear to a coffee date?",
        intent="style_advice",
    )
    assert coffee["mode"] == "style_advice"
    assert coffee["should_generate_board"] is False
    assert any(chip["label"] == "Use my wardrobe" for chip in coffee["cta"])
    assert len(coffee["visual_directions"]) == 3

    visual = style_reasoning_engine.reason(
        query="Show visual inspiration for coffee date",
        intent="style_advice",
    )
    assert visual["mode"] == "visual_inspiration"
    assert visual["should_generate_board"] is False
    assert len(visual["visual_directions"]) == 3

    wardrobe = style_reasoning_engine.reason(
        query="Use my wardrobe for a coffee date",
        intent="wardrobe_style",
    )
    assert wardrobe["mode"] == "wardrobe_style"
    assert wardrobe["should_use_wardrobe"] is True
    assert wardrobe["should_generate_board"] is True
    assert wardrobe["visual_directions"] == []

    color = style_reasoning_engine.reason(
        query="What colors suit warm olive skin?",
        intent="color_body_advice",
    )
    assert color["mode"] == "color_body_advice"
    assert color["should_generate_board"] is False
    assert len(color["visual_directions"]) == 3


def test_style_pairing_reasoning_returns_anchor_and_distinct_routes(monkeypatch):
    from services.style_reasoning_engine import style_reasoning_engine

    monkeypatch.setattr("services.style_reasoning_engine.generate_text", _fake_style_pairing_json)

    response = style_reasoning_engine.reason(
        query="What to pair with a white shirt?",
        intent="style_pairing",
    )

    assert response["mode"] == "style_pairing"
    assert response["should_generate_board"] is False
    assert response["should_use_wardrobe"] is False
    assert response["anchor_item"]["name"] == "white shirt"
    assert response["anchor_item"]["category"] == "shirt"
    assert response["anchor_item"]["color"] == "white"
    assert len(response["pairing_routes"]) >= 4
    assert len({route["title"] for route in response["pairing_routes"]}) >= 4
    assert len(response["visual_directions"]) == len(response["pairing_routes"])
    assert all(route.get("use_case") for route in response["pairing_routes"])
    assert all(route.get("archetype") for route in response["pairing_routes"])
    assert all(direction.get("archetype") for direction in response["visual_directions"])
    assert response["pairing_routes"][0]["archetype"] != response["pairing_routes"][0]["title"]


def test_style_pairing_fallback_for_black_loafers_has_mixed_use_cases(monkeypatch):
    from services.style_reasoning_engine import style_reasoning_engine

    def fail_generate(*args, **kwargs):
        raise RuntimeError("gemini offline")

    monkeypatch.setattr("services.style_reasoning_engine.generate_text", fail_generate)

    response = style_reasoning_engine.reason(
        query="How do I style black loafers?",
        intent="style_pairing",
    )

    assert response["mode"] == "style_pairing"
    assert response["anchor_item"]["category"] == "footwear"
    assert response["anchor_item"]["color"] == "black"
    titles = {route["title"] for route in response["pairing_routes"]}
    assert {"Smart Casual", "Office Clean", "Evening Minimal", "Weekend Neat"}.issubset(titles)


def test_stylist_advice_response_uses_gemini_visual_envelope(monkeypatch):
    from services.stylist_knowledge_service import build_stylist_advice_response

    monkeypatch.setattr("services.stylist_knowledge_service.generate_text", _fake_style_reasoning_json)

    response = build_stylist_advice_response(
        query="Show visual inspiration for a coffee date",
        mode="visual_inspiration",
        module_context="style",
    )

    assert response["success"] is True
    assert response["type"] == "stylist_advice"
    assert response["intent"] == "visual_inspiration"
    assert response["style_boards"] == []
    assert response["meta"]["mode"] == "stylist_knowledge_gemini"
    assert response["meta"]["wardrobe_lookup"] is False
    assert response["meta"]["goal"]
    assert response["meta"]["atmosphere"]
    assert len(response["data"]["visual_directions"]) == 3
    assert len(response["cards"]) == 3
    assert all(card["type"] == "visual_direction" for card in response["cards"])
    text = response["message_text"]
    assert "Styling principles" not in text
    assert "Outfit direction" not in text
    assert "Color harmony" not in text


def test_stylist_advice_response_fallback_is_not_robotic(monkeypatch):
    from services.stylist_knowledge_service import build_stylist_advice_response

    def fail_generate(*args, **kwargs):
        raise RuntimeError("gemini offline")

    monkeypatch.setattr("services.stylist_knowledge_service.generate_text", fail_generate)

    response = build_stylist_advice_response(
        query="What should I wear to a Christian funeral?",
        mode="style_advice",
        module_context="style",
    )

    assert response["intent"] == "style_advice"
    assert response["style_boards"] == []
    assert response["meta"]["wardrobe_lookup"] is False
    assert len(response["data"]["visual_directions"]) == 3
    assert "Styling principles" not in response["message_text"]
    assert "Outfit direction" not in response["message_text"]
    assert "Color harmony" not in response["message_text"]


def test_find_this_grounding_infers_plain_item_metadata():
    grounded = chat._find_this_grounding(
        "Find this: Dark Brown Penny Loafers",
        user_profile={"gender": "male"},
    )

    assert grounded["item_name"] == "Dark Brown Penny Loafers"
    assert grounded["category"] == "footwear"
    assert grounded["subcategory"] == "loafers"
    assert grounded["color"] == "dark brown"
    assert grounded["gender"] == "men"
    assert grounded["search_query"] == "Dark Brown Penny Loafers men"


def test_find_this_grounding_preserves_structured_metadata():
    response = chat._shopping_intent_response(
        "Find this: Dark Brown Penny Loafers | category=footwear | "
        "subcategory=loafers | color=dark brown | occasion=startup_office | "
        "archetype=Modern Operator",
        user_profile={"gender": "male"},
    )

    block = response["data"]["shopping_intent"]
    assert block["item_name"] == "Dark Brown Penny Loafers"
    assert block["category"] == "footwear"
    assert block["subcategory"] == "loafers"
    assert block["color"] == "dark brown"
    assert block["occasion"] == "startup_office"
    assert block["archetype"] == "Modern Operator"
    assert block["search_query"] == "Dark Brown Penny Loafers men"
    assert response["meta"]["find_this_grounded"] is True


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
    assert reject_board_for_occasion({"items": [shiny_gold_shirt]}, "office") in {
        "shiny_gold_shirt_blocked_for_smart_occasion",
        "metadata_v2.risky_item_for_professional_occasion",
    }

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


def test_adaptive_style_router_and_diet_guard():
    """Open-ended style questions must not misroute to diet when they mention a
    meal-time word; transition prompts route to multi-event style."""
    from services.style_context_service import detect_multi_event
    from routers import chat

    def _route(q):
        if detect_multi_event(q):
            return "transition_style"
        vb = chat._detect_visual_board_type(q, "")
        if vb == "diet_plan" and chat._is_style_priority_query(q):
            return "style"
        if vb == "diet_plan":
            return "diet"
        return "style"

    assert _route("How do I transition from a basketball game at 9pm to dinner at 11pm?") == "transition_style"
    assert _route("I have a basketball match then dinner, suggest outfit") == "transition_style"
    assert _route("Office meeting then drinks outfit") == "transition_style"
    assert _route("What should I eat for dinner?") == "diet"
    assert _route("Light dinner ideas") == "diet"
    assert _route("What should I wear for dinner?") == "style"


def test_style_priority_guard_helpers():
    from routers import chat

    assert chat._is_style_priority_query("what should I wear for dinner") is True
    assert chat._is_style_priority_query("how do I style black loafers") is True
    assert chat._is_style_priority_query("what should I avoid for a coffee date") is True
    assert chat._is_style_priority_query("how can I look taller") is True
    # explicit food beats style words
    assert chat._is_style_priority_query("light dinner ideas") is False
    assert chat._is_style_priority_query("what to eat for dinner") is False
