import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'VerificationType':2,'Amount':5,'Interest':5,'DebtToIncome':5,'PrincipalPaymentsMade':7,'AmountOfPreviousLoansBeforeLoan':7,
          'PreviousEarlyRepaymentsCountBeforeLoan':7,'LoanDuration':5,'NrOfScheduledPayments':5})

print(r.json())