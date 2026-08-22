"""Unit tests for Plan & Pack flow integration.
"""

import asyncio
import unittest

from brain.plan_pack_flow import build_plan_pack_response
from services.module_chat_service import handle_module_chat


class TestPlanPackFlow(unittest.TestCase):

    def test_build_plan_pack_response_carry_on(self):
        res = build_plan_pack_response("Pack for a carry-on trip")
        self.assertEqual(res["intent"], "plan_pack")
        self.assertIn("carry-on", res["message"].lower())
        self.assertEqual(res["type"], "checklists")
        self.assertEqual(res["visual_type"], "visual_packing_checklist")
        
        # Verify visual_sections array contains structured category groups
        visual_sections = res.get("visual_sections", [])
        self.assertTrue(len(visual_sections) >= 5)
        section_ids = [s["id"] for s in visual_sections]
        self.assertIn("clothes", section_ids)
        self.assertIn("essentials", section_ids)
        self.assertIn("tech", section_ids)
        self.assertIn("documents", section_ids)
        self.assertIn("weather", section_ids)

    def test_handle_module_chat_planner(self):
        payload = {
            "domain": "planner",
            "message": "Pack for a carry-on trip",
        }
        res = asyncio.run(handle_module_chat(payload, user_id="test_user"))
        self.assertTrue(res["success"])
        self.assertEqual(res["domain"], "planner")
        self.assertIn("cards", res)
        self.assertIn("visual_sections", res["data"])


if __name__ == "__main__":
    unittest.main()
