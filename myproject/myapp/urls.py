from django.urls import path 
from .views import home, showitems, professor_dropdown, professor_profile, search_prof, WriteReview, WriteReviewBlank, Databaseshow, delete_review, check_privacy_risk

urlpatterns = [
    path('', home, name='home'),
    #path('register/', register_view, name='register'),
    path('browse/', showitems, name='showitems'),
    path('search/', search_prof, name='search_prof'),
    path('professors/', professor_dropdown, name='professor_dropdown'),
    path('professor/<str:professor_name>/', professor_profile, name='professor_profile'),
    path('write/', WriteReviewBlank, name='WriteReviewBlank'),
    path('write/<str:professor_name>/', WriteReview, name='WriteReview'),
    path('datashow/', Databaseshow, name='Databaseshow'),
    path('review/<int:review_id>/delete/', delete_review, name='delete_review'),
    path('api/check-privacy-risk/', check_privacy_risk, name='check_privacy_risk'),
]
