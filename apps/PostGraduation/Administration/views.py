from rest_framework import viewsets
from . import models
from . import serializers


from Coord.client import coord
from e2e import basic


class DoctorantViewSet(viewsets.ModelViewSet):
    queryset = models.Doctorant.objects.all()
    serializer_class = serializers.DoctorantSerializer

    def create(self, request, *args, **kwargs):
        return basic.wrap('CreateDoctorant', super().create, request, *args, **kwargs)
    def update(self, request, *args, **kwargs):
        return basic.wrap('UpdateDoctorant', super().update, request, *args, **kwargs)
    def delete(self, request, *args, **kwargs):
        return basic.wrap('DeleteDoctorant', super().delete, request, *args, **kwargs)

class ModuleViewSet(viewsets.ModelViewSet):
    queryset = models.Module.objects.all()
    serializer_class = serializers.ModuleSerializer
    lookup_field = 'niveau'

    def create(self, request, *args, **kwargs):
        return basic.wrap('CreateModule', super().create, request, *args, **kwargs)
    def update(self, request, *args, **kwargs):
        return basic.wrap('UpdateModule', super().update, request, *args, **kwargs)
    def delete(self, request, *args, **kwargs):
        return basic.wrap('DeleteModule', super().delete, request, *args, **kwargs)

class RecoursViewSet(viewsets.ModelViewSet):
    queryset = models.Recours.objects.all()
    serializer_class = serializers.RecoursSerializer

    def create(self, request, *args, **kwargs):
        return basic.wrap('CreateRecours', super().create, request, *args, **kwargs)
    def update(self, request, *args, **kwargs):
        return basic.wrap('UpdateRecours', super().update, request, *args, **kwargs)
    def delete(self, request, *args, **kwargs):
        return basic.wrap('DeleteRecours', super().delete, request, *args, **kwargs)
    
class SujetViewSet(viewsets.ModelViewSet):
    queryset = models.Sujet.objects.all()
    serializer_class = serializers.SujetSerializer
    lookup_field = 'id'

    def create(self, request, *args, **kwargs):
        return basic.wrap('CreateSujet', super().create, request, *args, **kwargs)
    def update(self, request, *args, **kwargs):
        return basic.wrap('UpdateSujet', super().update, request, *args, **kwargs)
    def delete(self, request, *args, **kwargs):
        return basic.wrap('DeleteSujet', super().delete, request, *args, **kwargs)
    

class ReinscriptionViewSet(viewsets.ModelViewSet):
    queryset = models.Reinscription.objects.all()
    serializer_class = serializers.ReinscriptionSerializer

    def create(self, request, *args, **kwargs):
        return basic.wrap('CreateReinscription', super().create, request, *args, **kwargs)
    def update(self, request, *args, **kwargs):
        return basic.wrap('UpdateReinscription', super().update, request, *args, **kwargs)
    def delete(self, request, *args, **kwargs):
        return basic.wrap('DeleteReinscription', super().delete, request, *args, **kwargs)

class InscriptionViewSet(viewsets.ModelViewSet):
    queryset = models.Inscription.objects.all()
    serializer_class = serializers.InscriptionSerializer

    def create(self, request, *args, **kwargs):
        return basic.wrap('CreateInscription', super().create, request, *args, **kwargs)
    def update(self, request, *args, **kwargs):
        return basic.wrap('UpdateInscription', super().update, request, *args, **kwargs)
    def delete(self, request, *args, **kwargs):
        return basic.wrap('DeleteInscription', super().delete, request, *args, **kwargs)

class EnseignantViewSet(viewsets.ModelViewSet):
    queryset = models.Enseignant.objects.all()
    serializer_class = serializers.EnseignantSerializer

    def create(self, request, *args, **kwargs):
        return basic.wrap('CreateEnseignant', super().create, request, *args, **kwargs)
    def update(self, request, *args, **kwargs):
        return basic.wrap('UpdateEnseignant', super().update, request, *args, **kwargs)
    def delete(self, request, *args, **kwargs):
        return basic.wrap('DeleteEnseignant', super().delete, request, *args, **kwargs)

class PassageGradeViewSet(viewsets.ModelViewSet):
    queryset = models.PassageGrade.objects.all()
    serializer_class = serializers.PassageGradeSerializer

    def create(self, request, *args, **kwargs):
        return basic.wrap('CreatePassageGrade', super().create, request, *args, **kwargs)
    def update(self, request, *args, **kwargs):
        return basic.wrap('UpdatePassageGrade', super().update, request, *args, **kwargs)
    def delete(self, request, *args, **kwargs):
        return basic.wrap('DeletePassageGrade', super().delete, request, *args, **kwargs)
