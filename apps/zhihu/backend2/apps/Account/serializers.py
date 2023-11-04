import re
from datetime import datetime, timedelta

from rest_framework import serializers
from rest_framework.reverse import reverse
from rest_framework.validators import UniqueValidator

from django.contrib.auth import get_user_model
from .models import VerifyCode

User = get_user_model()


class UserProfileSerializer(serializers.ModelSerializer):

    """
    UserProfile Serializer
    """

    fav = serializers.StringRelatedField(many=True)
    password = serializers.CharField(
        style={'input_type': 'password'}, label="密码", write_only=True,
    )
    links = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = ("username", "gender", "email", "password", "views", "links", "fav")

    def get_links(self, obj):
        request = self.context['request']
        return reverse('user-detail', kwargs={'pk': obj.pk},
                       request=request)


def _regex_match(pat, s):
    from Analyzer.symbolic import Sym
    from Verifier.checker import RegexMatch
    if isinstance(s, Sym):
        return RegexMatch(pat, s.expr)
    else:
        return re.match(pat, s)


class SmsSerializer(serializers.Serializer):

    """
    验证码序列化
    """

    mobile = serializers.CharField(max_length=11)

    def validate_mobile(self, mobile):

        # 验证手机号码是否合法
        REGEX_MOBILE = "^1[358]\d{9}$|^147\d{8}$|^176\d{8}$"
        if not _regex_match(REGEX_MOBILE, mobile):
            raise serializers.ValidationError("请输入正确的手机号")

        # 手机号是否注册
        if User.objects.filter(mobile=mobile).count():
            raise serializers.ValidationError("您输入的手机号已注册,请登陆")

        # 验证发送频率
        one_mintes_ago = datetime.now() - timedelta(hours=0, minutes=1, seconds=0)
        if VerifyCode.objects.filter(created__gt=one_mintes_ago, mobile=mobile):
            raise serializers.ValidationError("验证频率过高，请一分钟后重试")

        return mobile


class UserRegisterSerializer(serializers.ModelSerializer):

    """
    用户注册序列化
    """

    mobile = serializers.CharField(
            label="手机号", required=True, allow_blank=False,
            validators=[UniqueValidator(queryset=User.objects.all(),
                                        message="手机号已注册")])

    code = serializers.CharField(required=True, write_only=True, max_length=4,
                                 min_length=4, label="验证码", error_messages={
                                    "blank": "请输入验证码",
                                    "required": "请输入验证码",
                                    "max_length": "验证码格式错误",
                                    "min_length": "验证码格式错误"
                                 })
    username = serializers.CharField(
            label="用户名", required=True, allow_blank=False,
            validators=[UniqueValidator(queryset=User.objects.all(),
                                        message="用户名已存在")])
    password = serializers.CharField(
            min_length=6, label="密码", write_only=True,
            style={'input_type': 'password'},
            error_messages={
                "min_length": "密码最少为6位"
            }
            )

    def validate_code(self, code):
        verify_records = VerifyCode.objects.filter(
                mobile=self.initial_data["mobile"]
            ).order_by("-created")
        if verify_records:
            last_record = verify_records[0]
            five_mintes_ago = datetime.now() - timedelta(hours=0, minutes=5,
                                                         seconds=0)
            if five_mintes_ago > last_record.created:
                raise serializers.ValidationError("验证码已过期")
            if last_record.code != code:
                raise serializers.ValidationError("验证码错误")
        else:
            raise serializers.ValidationError("验证码错误")

    # NOTE(ksqsf): see Notes in `views.UserRegisterViewSet`
    def validate(self, attrs):
        # FIXME(ksqsf): see patch_drf.Data, it ignores such deletion. But
        # should it?
        del attrs["code"]
        return attrs

    def create(self, validated_data):
        user = super(UserRegisterSerializer, self).create(validated_data=validated_data)
        user.set_password(validated_data["password"])
        user.save()
        return user

    class Meta:
        model = User
        fields = ("mobile", "code", "username", "password")
