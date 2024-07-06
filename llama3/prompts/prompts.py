class PromptCollection():
    def __init__(self) -> None:
        pass

    def date_prompt(self, patient_chart):
        prompt = f"""
You are a medical curator and you are extracting all kinds of dates from below charts to hide personal information (Deidentification of medical charts). Extract the dates and give an output in the form of a json.

Things to keep in mind while extracting dates:

1. Consider dates without year or even a month as well
2. If a date is repeated in different form extract that as well
3. Extract them exactly as mentioned in the chart, do not modify, for example, if a date is written as 1st of April, extract `1st of April`, don't change it to date format
4. Don't add any explanation or any other text

Json output should look like this:

{{
  "dates":['date 1', 'date 2', ...]
}}

Patient chart is given below in backticks.
```
{patient_chart}
```
"""
        return prompt