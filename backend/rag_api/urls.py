from django.urls import path
from .views import ask

urlpatterns = [
    # Mapping post requests to the logic controller
    path("ask/", ask, name="ask-question"),
]