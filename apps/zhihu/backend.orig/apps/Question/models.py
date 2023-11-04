from django.db import models
from django.utils import timezone
from django.contrib.auth import get_user_model
from django.db import IntegrityError

User = get_user_model()


class Topic(models.Model):

    """
    话题模型
    """

    TYPE_CHOICES = (
        (1, "一级话题"),
        (2, "二级话题"),
        (3, "三级话题"),
    )

    name = models.CharField(max_length=150, verbose_name="话题名")
    desc = models.TextField(verbose_name="话题描述")
    parent_topic = models.ForeignKey('self', null=True, blank=True,
                                     related_name='child', verbose_name="父话题",
                                     on_delete=models.CASCADE)
    topic_type = models.IntegerField(choices=TYPE_CHOICES, verbose_name="话题级别")
    flows = models.PositiveIntegerField(default=0, verbose_name="关注")

    class Meta:
        verbose_name = "话题"
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.name


class Question(models.Model):

    """
    问题模型
    """

    title = models.CharField(max_length=150, verbose_name="标题")
    body = models.TextField(verbose_name="内容", null=True, blank=True)
    created = models.DateTimeField(auto_now_add=True, verbose_name="创建日期")
    updated = models.DateTimeField(auto_now=True, verbose_name="更新日期")
    views = models.PositiveIntegerField(default=0)
    author = models.ForeignKey(User, related_name='questions',
                               verbose_name="提问者", on_delete=models.CASCADE,)
    topic = models.ForeignKey(Topic, related_name='questions', blank=True,
                              verbose_name="话题", on_delete=models.CASCADE,
                              null=True,)
    anonymous = models.BooleanField(default=False, verbose_name='是否匿名')
    flows = models.PositiveIntegerField(default=0, verbose_name="关注")

    class Meta:
        ordering = ('-updated',)
        verbose_name = "问题"
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.title

    def increase_views(self):
        self.views += 1
        self.save(update_fields=['views'])

    def flow(self):
        try:
            self.flows += 1
            self.save()
        except IntegrityError:
            return 'already_flowed'
        return 'ok'

    def cancel_flow(self):
        try:
            self.flows -= 1
            self.save()
        except IntegrityError:
            return 'already_cancel'
        return 'ok'


class Answer(models.Model):

    """
    回答模型
    """

    STATUS_CHOICES = (
        ('draft', '草稿'),
        ('published', '已发布')
    )

    author = models.ForeignKey(User, related_name='answers',
                               verbose_name="回答者", on_delete=models.CASCADE)
    question = models.ForeignKey(Question, related_name="answers",
                                 verbose_name="问题", on_delete=models.CASCADE)
    text = models.TextField(verbose_name="内容")
    publish = models.DateTimeField(default=timezone.now, verbose_name="发布日期")
    created = models.DateTimeField(auto_now_add=True, verbose_name="创建日期")
    updated = models.DateTimeField(auto_now=True, verbose_name="更新日期")
    status = models.CharField(max_length=10, choices=STATUS_CHOICES,
                              default='draft', verbose_name="回答状态")
    vote = models.IntegerField(default=0, verbose_name="赞同数")
    collection = models.ForeignKey(User, verbose_name="收藏", blank=True,
                                   null=True, on_delete=models.CASCADE,
                                   related_name="collection")
    anonymous = models.BooleanField(default=False, verbose_name='是否匿名')
    flows = models.PositiveIntegerField(default=0, verbose_name="关注")

    # 赞同问题
    def up_vote(self):
        try:
            self.vote += 1
            self.save()
        except IntegrityError:
            return 'already_upvoted'
        return 'ok'

    # 反对问题
    def down_vote(self):
        try:
            self.vote -= 1
            self.save()
        except IntegrityError:
            return 'already_downvoted'
        return 'ok'

    class Meta:
        ordering = ('-vote',)
        verbose_name = "回答"
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.text


class Comment(models.Model):

    """
    回答评论
    """

    pass
