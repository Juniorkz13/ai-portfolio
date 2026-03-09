from app.main import app, create_app


def test_app_registers_core_routes():
    route_paths = {route.path for route in app.routes}

    assert "/health" in route_paths
    assert "/upload" in route_paths
    assert "/chat" in route_paths


def test_create_app_returns_fastapi_instance_with_health_route():
    created_app = create_app()
    route_paths = {route.path for route in created_app.routes}

    assert "/health" in route_paths
