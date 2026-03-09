from app.core.database import Base, engine
import app.models  # noqa: F401


if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")
