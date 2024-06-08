from django.db import models

# Create your models here.
class CutFreeModel:
    def __init__(self):
        self.model = None

    def predict(self, starting_oligo, restriction_sites):
        # predict
        # result = self.model.predict(starting_oligo, restriction_sites)
        return "cutfree"