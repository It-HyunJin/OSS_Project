from django import forms

class PredictForm(forms.Form):
    text = forms.CharField(widget=forms.Textarea, label='텍스트 입력')
