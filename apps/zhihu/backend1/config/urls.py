from django.contrib import admin
from django.conf.urls import include
from django.urls import path, re_path

from rest_framework.routers import DefaultRouter
from rest_framework_jwt.views import obtain_jwt_token
from rest_framework.authtoken import views

from Question.views import AnswerViewset, TopicViewset, QuestionViewset
from Account.views import UserProfileViewSet, SmsCodeViewSet, UserRegisterViewset
from AccountOperation.views import UserVoteViewSet, UserFlowQuestionViewSet, UserFavViewSet


router = DefaultRouter()

router.register(r'answers', AnswerViewset, basename='answer')
router.register(r'topics', TopicViewset, basename='topic')
router.register(r'questions', QuestionViewset, basename='question')
router.register(r'users', UserProfileViewSet, basename='user')
router.register(r'codes', SmsCodeViewSet, basename='code')
router.register(r'register', UserRegisterViewset, basename='register')
router.register(r'votes', UserVoteViewSet, basename='vote')
router.register(r'flow_questions', UserFlowQuestionViewSet, basename='flow_question')
router.register(r'favs', UserFavViewSet, basename='fav')

urlpatterns = [
    # ADMIN URL
    path('admin/', admin.site.urls),
    # API LOGIN
    path('api/v1/login/', obtain_jwt_token),
    # API URL
    re_path('api/v1/', include(router.urls)),
    # re_path('api/v2/', include(router.urls)),
    # re_path('api/v3/', include(router.urls)),
    # DRF LOGIN
    path('api-token-auth/', views.obtain_auth_token),
    re_path('api-auth/', include('rest_framework.urls', namespace='rest_framework')),
]
