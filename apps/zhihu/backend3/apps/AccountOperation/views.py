from rest_framework import mixins
from rest_framework import viewsets
from rest_framework_jwt.authentication import JSONWebTokenAuthentication
from rest_framework.authentication import SessionAuthentication
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import filters

from .serializers import UserVoteSerializer, UserFlowQuestionSerializer, UserFavSerializer
from .models import UserVote, UserFlowQuestion, UserFav

# NOTE: only for E2E
from Coord.client import coord
from e2e import basic

class UserVoteViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet,
                      mixins.CreateModelMixin, mixins.DestroyModelMixin):

    """
    用户赞同 / 反对
    """

    serializer_class = UserVoteSerializer
    authentication_classes = (JSONWebTokenAuthentication, SessionAuthentication)
    filter_backends = (DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter,)
    filter_fields = ('vote_type',)
    search_fields = ('answer__question__id',)
    lookup_field = "answer"

    def get_queryset(self):
        # FIXME(ksqsf) both self.request and self.request.user are concrete values
        # TODO(ksqsf) This seems to be fixed. Confirm later.
        return UserVote.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        instance = serializer.save()
        vote_type = instance.vote_type
        answer = instance.answer
        if vote_type == 'up':
            answer.up_vote()
        if vote_type == 'down':
            answer.down_vote()

    def perform_destroy(self, instance):
        vote_type = instance.vote_type
        answer = instance.answer
        if vote_type == 'up':
            answer.down_vote()
        if vote_type == 'down':
            answer.up_vote()
        instance.delete()

    # NOTE: only for E2E
    def create(self, request, *args, **kwargs):
        return basic.wrap('CreateUserVote', super().create, request, *args, **kwargs)
    def delete(self, request, *args, **kwargs):
        return basic.wrap('DeleteUserVote', super().delete, request, *args, **kwargs)

class UserFlowQuestionViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin,
                              mixins.CreateModelMixin, mixins.DestroyModelMixin,
                              viewsets.GenericViewSet):

    """
    用户关注问题
    """

    serializer_class = UserFlowQuestionSerializer
    authentication_classes = (JSONWebTokenAuthentication, SessionAuthentication)
    lookup_field = "question"

    def get_queryset(self):
        return UserFlowQuestion.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        instance = serializer.save()
        question = instance.question
        question.flow()

    def perform_destroy(self, instance):
        question = instance.question
        question.cancel_flow()
        question.delete()
        instance.delete()

    # NOTE: only for E2E
    def create(self, request, *args, **kwargs):
        return basic.wrap('CreateUserFlowQuestion', super().create, request, *args, **kwargs)
    def delete(self, request, *args, **kwargs):
        return basic.wrap('DeleteUserFlowQuestion', super().delete, request, *args, **kwargs)


class UserFavViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin,
                     mixins.CreateModelMixin, mixins.DestroyModelMixin,
                     viewsets.GenericViewSet):

    """
    用户收藏回答
    """

    authentication_classes = (JSONWebTokenAuthentication, SessionAuthentication)
    serializer_class = UserFavSerializer
    lookup_field = "answer"

    def get_queryset(self):
        return UserFav.objects.filter(user=self.request.user)

    # def perform_destroy(self, instance):
    #     print('destroy ksqsf:', instance, type(instance), instance.expr)
    #     instance.delete()

    # NOTE: only for E2E
    def create(self, request, *args, **kwargs):
        return basic.wrap('CreateUserFav', super().create, request, *args, **kwargs)
    def delete(self, request, *args, **kwargs):
        return basic.wrap('DeleteUserFav', super().delete, request, *args, **kwargs)
