from datetime import date

from sqlalchemy import Date, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Book(Base):
    __tablename__ = "books"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    author: Mapped[str] = mapped_column(String(255))
    published_date: Mapped[date] = mapped_column(Date)
    summary: Mapped[str] = mapped_column(Text)
