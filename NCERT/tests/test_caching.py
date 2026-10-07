import unittest
import os
import shutil
from app.smart_cache import SmartCache, extract_numbers_and_units, extract_key_entities

class TestSmartCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db = "data/unit_test_cache.db"
        if os.path.exists(cls.test_db):
            os.remove(cls.test_db)
        cls.cache = SmartCache(cls.test_db)

        # Seed with initial queries
        cls.cache.store(
            "What is refraction?",
            "Refraction is the phenomenon of bending of light rays when they pass from one transparent medium to another.",
            ["Light – Reflection and Refraction"]
        )
        cls.cache.store(
            "Image formed by a concave mirror when object is at infinity",
            "When the object is at infinity, the image is formed at focus F, real and inverted, highly diminished point-sized.",
            ["Light – Reflection and Refraction"]
        )
        cls.cache.store(
            "Calculate focal length when R = 20 cm",
            "Given R = 20 cm, focal length f = R / 2 = 20 / 2 = 10 cm.",
            ["Light – Reflection and Refraction"]
        )

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_db):
            os.remove(cls.test_db)

    def test_same_doubt_different_wording(self):
        """Situation 1: Same doubt, different wording -> Serve from cache (HIT)"""
        result = self.cache.lookup("What does refraction mean?")
        self.assertIsNotNone(result)
        reply, citations = result
        self.assertIn("Light – Reflection and Refraction", citations)
        self.assertIn("bending of light", reply)

    def test_similar_words_different_entity(self):
        """Situation 2: Similar words, different entity (concave vs convex) -> DO NOT serve (MISS)"""
        result = self.cache.lookup("Image formed by a convex mirror when object is at infinity")
        self.assertIsNone(result)

    def test_same_question_different_numbers(self):
        """Situation 3: Same question, different numbers (R = 20 cm vs R = 30 cm) -> DO NOT serve (MISS)"""
        result = self.cache.lookup("Calculate focal length when R = 30 cm")
        self.assertIsNone(result)

    def test_followup_pronoun_laws(self):
        """Situation 4: Follow-up that depends on earlier turns -> Unresolved pronoun -> NEVER direct cache hit"""
        result = self.cache.lookup("What about its laws?")
        self.assertIsNone(result)

    def test_conversation_tied_request(self):
        """Situation 5: Request tied to this conversation ('Explain it more simply') -> NEVER serve from cache"""
        result = self.cache.lookup("Explain it more simply")
        self.assertIsNone(result)

        result2 = self.cache.lookup("Can you simplify that?")
        self.assertIsNone(result2)

if __name__ == "__main__":
    unittest.main()
