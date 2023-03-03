from django import forms 

class ParamsApriori(forms.Form):
    support=forms.FloatField(label="support", required=True)
    confidence=forms.FloatField(label="confidence",required=True)
    lift= forms.FloatField(label="lift",required=True)