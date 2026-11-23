import unittest

from contact_monitor import ContactHold


class ContactHoldTest(unittest.TestCase):
    def test_latches_at_threshold(self):
        monitor = ContactHold(5.0)
        self.assertFalse(monitor.update([4.9, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.assertTrue(monitor.update([5.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    def test_remains_latched_after_force_drops(self):
        monitor = ContactHold(5.0)
        monitor.update([6.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertTrue(monitor.update([0.0] * 6))
        self.assertEqual(monitor.detected_force_n, 6.0)


if __name__ == "__main__":
    unittest.main()
