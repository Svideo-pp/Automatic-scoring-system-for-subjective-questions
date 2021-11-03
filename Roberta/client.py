import requests

# sa=Operator is a special symbol used to operate a data value.&ra=Operators function to manipulate the value of a variable.
info = {'sa': 'Operator is a special symbol used to operate a data value.',
        'ra': 'Operators function to manipulate the value of a variable.'}

r = requests.post('http://127.0.0.1:5000/auto_grading', data=info)
print(r.text)

