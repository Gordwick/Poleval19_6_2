import pytest
from fastapi.testclient import TestClient

from run_server import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
        

def test_main_predict(client):
    """
    Test predction response
    """

    headers = {}
    body = {
        "message": "Ala ma kota."
    }

    response = client.post("/api/v1/predict",
                           headers=headers,
                           json=body)

    try:
        assert response.status_code == 200
        reponse_json = response.json()
        assert reponse_json['error'] == False
        assert isinstance(reponse_json['results']['prediction'], str)

    except AssertionError:
        print(response.status_code)
        print(response.json())
        raise