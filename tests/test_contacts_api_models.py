from models.ahvi_contact_models import contact_to_appwrite


def test_contact_mapper_accepts_legacy_name_phone_payload():
    payload = contact_to_appwrite(
        {
            "name": "Ravi Kumar",
            "phone": 6305685757,
            "relationship": "Family",
            "tags": [],
            "avatarUrl": "",
            "notes": "",
            "favorite": True,
        },
        "user_1",
    )

    assert payload["userId"] == "user_1"
    assert payload["firstName"] == "Ravi"
    assert payload["lastName"] == "Kumar"
    assert payload["phoneNumber"] == "6305685757"
    assert payload["isFavorite"] is True


def test_contact_mapper_preserves_current_contact_fields():
    payload = contact_to_appwrite(
        {
            "firstName": "Meera",
            "lastName": "Rao",
            "phoneNumber": "9876543210",
            "isFavorite": False,
        },
        "user_1",
    )

    assert payload["firstName"] == "Meera"
    assert payload["lastName"] == "Rao"
    assert payload["phoneNumber"] == "9876543210"
