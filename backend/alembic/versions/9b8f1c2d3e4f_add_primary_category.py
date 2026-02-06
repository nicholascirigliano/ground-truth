"""add primary_category to articles

Revision ID: 9b8f1c2d3e4f
Revises: c176427dc610
Create Date: 2026-02-05 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "9b8f1c2d3e4f"
down_revision = "c176427dc610"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("articles", sa.Column("primary_category", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("articles", "primary_category")
