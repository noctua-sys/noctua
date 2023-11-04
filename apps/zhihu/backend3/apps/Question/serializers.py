from rest_framework import serializers
from rest_framework.reverse import reverse

from .models import Answer, Topic, Question


class AnswerSerializer(serializers.ModelSerializer):

    """
    Answer Serializer
    """

    question_title = serializers.CharField(source='question.title', read_only=True)
    question_id = serializers.CharField(source='question.id', read_only=True)
    topic = serializers.CharField(source='question.topic', read_only=True)
    author_name = serializers.CharField(source='author.username', read_only=True)
    author_desc = serializers.CharField(source='author.desc', read_only=True)
    author = serializers.HiddenField(
        default=serializers.CurrentUserDefault(),
        write_only=True
    )
    links = serializers.SerializerMethodField(read_only=True)
    flows = serializers.CharField(read_only=True)
    collection = serializers.CharField(read_only=True)
    vote = serializers.CharField(read_only=True)

    class Meta:
        model = Answer
        fields = '__all__'

    def get_links(self, obj):
        request = self.context['request']
        return reverse('answer-detail', kwargs={'pk': obj.pk},
                       request=request)


class TopicSerializer(serializers.ModelSerializer):

    """
    Topic Serializer
    """

    class Meta:
        model = Topic
        fields = '__all__'


class QuestionSerializer(serializers.ModelSerializer):

    """
    Question Serializer
    """

    answers = AnswerSerializer(many=True, read_only=True)
    answers_count = serializers.SerializerMethodField(read_only=True)
    topic_name = serializers.CharField(source='topic.name', allow_blank=True, allow_null=True,
                                       read_only=True)
    author_name = serializers.CharField(source='author.username', read_only=True)
    author = serializers.HiddenField(
        default=serializers.CurrentUserDefault(),
        write_only=True
    )
    links = serializers.SerializerMethodField(read_only=True)
    views = serializers.CharField(read_only=True)

    class Meta:
        model = Question
        fields = '__all__'

    def get_links(self, obj):
        request = self.context['request']
        return reverse('question-detail', kwargs={'pk': obj.pk},
                       request=request)

    def get_answers_count(self, obj):
        return obj.answers.count()
