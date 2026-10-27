"""Order lookup and ticket creation tool wrapper for ZentroMart VoiceOps.

Gated behind caller identity verification:
- Step 1: lookup_order(order_id) -> returns verification prompt.
- Step 2: lookup_order(order_id, verification_value=...) -> phone last 4 digits OR delivery pincode.
- Maximum 1 retry allowed on failure.
- Never reveals expected proof or sensitive customer data without verification.
"""
import os
import sys
from typing import Dict, Optional

# Add dataset folder to sys.path to import original offline tools
_DATASET_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Capstone_Project_Dataset_FTTT_2026")
)
if _DATASET_ROOT not in sys.path:
    sys.path.insert(0, _DATASET_ROOT)

from tools.tools import lookup_order as _raw_lookup_order
from tools.tools import create_ticket as _raw_create_ticket

class OrderToolManager:
    def __init__(self):
        self.failed_attempts: Dict[str, int] = {}

    def lookup_order(self, order_id: str, verification_value: Optional[str] = None) -> Dict:
        """Call synthetic lookup_order tool with attempt tracking."""
        clean_oid = "".join(ch for ch in str(order_id) if "0" <= ch <= "9")
        
        if not verification_value:
            return _raw_lookup_order(clean_oid)

        clean_val = "".join(ch for ch in str(verification_value) if "0" <= ch <= "9")
        res = _raw_lookup_order(clean_oid, verification_value=clean_val)
        
        if not res.get("verified"):
            attempts = self.failed_attempts.get(clean_oid, 0) + 1
            self.failed_attempts[clean_oid] = attempts
            res["remaining_attempts"] = max(0, 2 - attempts)
            res["locked"] = attempts >= 2
        else:
            self.failed_attempts[clean_oid] = 0

        return res

    def create_ticket(self, summary: str, order_id: Optional[str] = None,
                      reason: Optional[str] = None, intent: Optional[str] = None) -> Dict:
        """Call synthetic create_ticket tool."""
        return _raw_create_ticket(summary=summary, order_id=order_id, reason=reason, intent=intent)
