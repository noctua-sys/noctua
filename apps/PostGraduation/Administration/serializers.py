from rest_framework import serializers
from . import models


class RecoursSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Recours
        fields = ('doctorant','sujet','message','accepted')

class PassageGradeSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.PassageGrade
        fields = ('id','enseignant','gradeVoulu','argument')

class ReinscriptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Reinscription
        fields = ('doctorant','intitulerPostGrade','intitulerSujet','diplomeGraduation','nomEncadreur','nomCoEncadreur','dateReinscription')

class InscriptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Inscription
        fields = ('doctorant','intitulerPostGrade','intitulerSujet','diplomeGraduation','nomEncadreur','nomCoEncadreur','dateInscription','gradeEncadreur','gradeCoEncadreur')

class EnseignantSerializer(serializers.ModelSerializer):
    passagegrades = serializers.HyperlinkedRelatedField(
        many=True,
        read_only=True,
        view_name='passagegrade-detail'
    )
    class Meta:
        model = models.Enseignant
        fields = ('id','nom','prenom','sexe' ,'date_naissance', 'lieu_naissance', 'addresse','email','grade','passagegrades')

class DoctorantSerializer(serializers.ModelSerializer):
    recours = serializers.HyperlinkedRelatedField(
        many=True,
        read_only=True,
        view_name='recours-detail'
    )
    reinscriptions = serializers.HyperlinkedRelatedField(
        many=True,
        read_only=True,
        view_name='reinscription-detail'
    )
    inscriptions = serializers.HyperlinkedRelatedField(
        many=True,
        read_only=True,
        view_name='inscription-detail'
    )

    class Meta:
        model = models.Doctorant
        fields = ('id','nationaliter', 'nom','prenom','sexe' ,'date_naissance', 'lieu_naissance', 'addresse','email','accepted','password','inscriptions','reinscriptions','recours')

class ModuleSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = models.Module
        fields = ('id','nom','niveau')

class SujetSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = models.Sujet
        fields = ('id','titre','description','accepted')

class EnseignantSerializer(serializers.HyperlinkedModelSerializer):
    passagegrades = serializers.HyperlinkedRelatedField(
        many=True,
        read_only=True,
        view_name='passagegrade-detail'    
    )
    class Meta:
        model = models.Enseignant
        fields = ('id', 'nom','prenom','sexe' ,'date_naissance', 'lieu_naissance', 'addresse','email','password','grade','passagegrades')
    