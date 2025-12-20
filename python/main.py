from classes import Category,EmbeddingService, MailClassyfire

categories = [
    Category(name="Финансы"),
    Category(name="Юридические вопросы"),
    Category(name="Техническая поддержка"),
    Category(name="Бизнес"),
    Category(name="Вакансия"),
    Category(name="Промо")
    # Либо любая категория, которая была введена пользователем
]

embedder = EmbeddingService()

results = []

mc = MailClassyfire()

different_emails = []

for email in different_emails:
  result = mc.classify_email(
      email_text=email["full_text"],
      categories=categories,
      embedder=embedder
  )

  results.append(result)