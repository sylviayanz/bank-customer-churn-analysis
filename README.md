# Bank Customer Churn Analysis

In the realm of credit card services, Customer Churn, often denoted as the Churn Rate or simply 'Churn', represents the proportion of customers who discontinue using a company's credit card products within a defined timeframe. 
Conceptually, the Churn Rate signifies the percentage of customers a company 'loses' over a specified period, and is occasionally termed the 'Attrition Rate'.

## Thought Process

- **Identify Valuable Customers**: Determine which groups are our most valuable customers, understand their backgrounds, and assess their satisfaction levels with our service.
- **Churn Rate Analysis**: Analyze the churn rate across different customer groups to identify patterns and correlations between churn rate and customer behavior.

## Dataset

We delve into a dataset of credit card customers to analyze the churn rate and discern potential correlations between churn rate and customer behavior.

## Key Insights

Our analysis reveals key insights into the churn rate among our customer base. Overall, the churn rate stands at 20.38%.

### Demographic Insights
- The churn rate is significantly higher among females compared to males.
- Our customer base is predominantly aged 25-45. However, the highest churn rates are observed in the 45-65 age group, suggesting our products may not appeal to the late middle-aged demographic or that competitors offer more attractive alternatives.

### Geographic Insights
- Our customers are primarily from France. However, the churn rate among German customers is notably higher, suggesting potential service issues in Germany that need addressing.

### Customer Complaints
- An alarming 99.8% of customers who eventually left our bank had filed complaints. This indicates a severe issue with how complaints are managed and resolved, necessitating an overhaul of our customer service approach to provide satisfactory solutions.

### Membership Activity
- Active members tend to stay with our bank more than inactive members.

### Product Analysis
- Customers with 3-4 products exhibit a high churn rate, particularly those with 4 products, who have a 100% churn rate. This suggests that these customers may be exploiting our offerings to meet bonus criteria and subsequently leaving.
- Customers with fewer than 200 reward points tend to leave, indicating they may not perceive sufficient benefits from our bank.

### High Balance Customers
- Customers with balances exceeding 20K have a high churn rate. This suggests that other banks may be offering more attractive incentives to high-balance customers.

## Recommendations

This analysis underscores the importance of understanding the factors driving customer churn and implementing targeted strategies to mitigate these issues. By focusing on customer satisfaction, service quality, and tailored product offerings, we can enhance customer loyalty and reduce churn.

It is well known that acquiring a new client is much more expensive than retaining an existing one. Therefore, it is advantageous for banks to understand the factors leading a client to decide to leave the company.

## Conclusion

By identifying the most valuable customers, analyzing the churn rate, and understanding customer behavior, we can develop effective strategies to improve customer retention and reduce churn.



##  The Dataset Contained the Following Information:

'RowNumber': A sequential index assigned to each row
'CustomerId': A unique serial key for each customer
'Surname': The customer's surname
'CreditScore': The customer's credit score
'Geography': The country of the customer
'Gender': The gender of the customer
'Age': The age of the customer
'Tenure': The number of years the customer has been with the bank
'Balance': The current balance in the customer's account
'NumOfProducts': The number of banking products the customers has with the bank
'HasCrCard': Whether or not the customer has a credit card with the bank
'IsActiveMember': Whether or not the customer is an active member of the bank = 'EstimatedSalary': The estimated salary of the customer
'EstimatedSalary': The estimated salary of the customer
'Exited': Whether or not the customer has churned (1 if they did, 0 if they did not)
'Exited': Whether or not the customer has churned (1 if they did, 0 if they did not)
'Complain': Whether or not the customer has filed a complaint
'SatisfactionScore': The customer's satisfaction score
'CardType': The type of credit card the customer has
'PointEarned': The number of reward points the customer has earned    

