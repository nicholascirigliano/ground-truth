from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    url: Mapped[str] = mapped_column(String, nullable=False)
    rss_url: Mapped[str] = mapped_column(String, nullable=False)
    default_category: Mapped[str] = mapped_column(String, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.now())

    articles: Mapped[list["Article"]] = relationship("Article", back_populates="source")


class Article(Base):
    __tablename__ = "articles"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    canonical_url: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    source_id: Mapped[str] = mapped_column(String, ForeignKey("sources.id"), nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.now())

    source: Mapped["Source"] = relationship("Source", back_populates="articles")
    summary: Mapped["Summary"] = relationship("Summary", back_populates="article", uselist=False)


class Summary(Base):
    __tablename__ = "summaries"

    article_id: Mapped[str] = mapped_column(String, ForeignKey("articles.id"), primary_key=True)
    summary_text: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.now())

    article: Mapped["Article"] = relationship("Article", back_populates="summary")
