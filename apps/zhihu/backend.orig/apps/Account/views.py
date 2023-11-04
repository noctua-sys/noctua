from random import choice

from django.contrib.auth.backends import ModelBackend
from django.contrib.auth import get_user_model
from django.db.models import Q

from rest_framework import mixins
from rest_framework import viewsets
from rest_framework import status
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response
from rest_framework_jwt import utils
# from rest_framework_jwt.serializers import jwt_encode_handler, jwt_payload_handler

from .serializers import UserProfileSerializer, SmsSerializer, UserRegisterSerializer
from utils.sms import YunPian
from config.settings import SMS_KEY
from .models import VerifyCode

from Analyzer import is_being_analyzed

User = get_user_model()


class CustomBackend(ModelBackend):

    """
    自定义用户验证
    """

    def authenticate(self, username=None, password=None, **kwargs):
        try:
            user = User.objects.get(Q(username=username) | Q(email=username) | Q(mobile=username))
            if user.check_password(password):
                return user
        except Exception as e:
            return None


class BasePagination(PageNumberPagination):

    """
    分页
    """

    page_size = 10
    page_size_query_param = 'page_size'
    max_page_size = 100


class UserProfileViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet):

    """
    用户
    """

    serializer_class = UserProfileSerializer
    pagination_class = BasePagination

    def get_queryset(self):
        return User.objects.filter(username=self.request.user.username)


class SmsCodeViewSet(mixins.CreateModelMixin, viewsets.GenericViewSet):

    """
    短信验证码生成
    """
    serializer_class = SmsSerializer

    def generate_code(self):
        """
        生成四位数字的验证码
        :return:
        """
        seeds = "1234567890"
        random_str = []
        for i in range(4):
            random_str.append(choice(seeds))

        return "".join(random_str)

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        mobile = serializer.validated_data["mobile"]
        yun_pian = YunPian(SMS_KEY)
        code = self.generate_code()

        if is_being_analyzed():
            sms_status = {'code': 0}
        else:
            sms_status = yun_pian.send_sms(code=code, mobile=mobile)

        if sms_status["code"] != 0:
            return Response({
                "mobile": sms_status["msg"]
            }, status=status.HTTP_400_BAD_REQUEST)
        else:
            code_record = VerifyCode.objects.create(code=code, mobile=mobile)
            code_record.save()
            return Response({
                "mobile": mobile
            }, status=status.HTTP_201_CREATED)


class UserRegisterViewset(mixins.CreateModelMixin, mixins.UpdateModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet):

    """
    用户注册
    """

    serializer_class = UserRegisterSerializer
    queryset = User.objects.all()

    # NOTE(ksqsf): update, see drf.mixins.UpdateModelMixin
    # 1. instance = drf.generics.GenericAPIView.get_object(self)
    # 2. serializer = drf.generics.GenericAPIView.get_serializer(self)
    # 3. drf.serializers.BaseSerializer.is_valid(serializer) -> drf.serializers.BaseSerializer.run_validation(serializer)
    #    # serializer.initial_data, initialized from cls init args
    #    # ... which is initialized from `request.data`
    #    - drf.serializers.BaseSerializer.run_validation
    #      - drf.fields.Field.run_validators
    #      - drf.serializer.Serializer.validate

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = self.perform_create(serializer)
        re_dict = serializer.data
        payload = utils.jwt_payload_handler(user)
        re_dict["token"] = utils.jwt_encode_handler(payload)
        re_dict["username"] = user.username
        headers = self.get_success_headers(serializer.data)
        return Response(re_dict, status=status.HTTP_201_CREATED, headers=headers)

    def perform_create(self, serializer):
        return serializer.save()
