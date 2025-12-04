#!/usr/bin/env python
"""Test import of super_admin_routes."""
import sys
sys.path.insert(0, 'd:\\School project\\EDINAI\\EDinai-Backend\\backend')

try:
    from app.routes.super_admin_routes import router
    print("SUCCESS: Router imported successfully")
    print(f"Router routes: {router.routes}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
