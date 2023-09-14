from sqlalchemy import Column, Integer, String, Text, ForeignKey, LargeBinary, Table, Float, UniqueConstraint
from sqlalchemy.orm import relationship
from .base import Base


class User(Base):
    __tablename__ = "User"

    Username = Column(String(255), primary_key=True, unique=True, nullable=False)
    Email = Column(String(255), unique=True, nullable=False)
    Password = Column(String(255), nullable=False)

    projects = relationship("Project", back_populates="user")
    #sentences = relationship("Sentence", back_populates="user")


class Dataset(Base):
    __tablename__ = "Dataset"

    DatasetID = Column(Integer, primary_key=True)
    ProjectID = Column(Integer, ForeignKey('Project.ProjectID', ondelete="CASCADE"))
    DatasetName = Column(String(255), nullable=False)
    Description = Column(Text)

    project = relationship("Project", back_populates="datasets")
    sentences = relationship("Sentence", back_populates="dataset")

class Project(Base):
    __tablename__ = "Project"

    ProjectID = Column(Integer, primary_key=True)
    Username = Column(String(255), ForeignKey('User.Username', ondelete="CASCADE"))
    ProjectName = Column(String(255), nullable=False)
    Description = Column(Text)

    user = relationship("User", back_populates="projects")
    datasets = relationship("Dataset", back_populates="project")
    annotations = relationship("Annotation", back_populates="project")
    views = relationship("View", back_populates="project")
    combined_models = relationship("CombinedModel", back_populates="project")


class Sentence(Base):
    __tablename__ = "Sentence"

    SentenceID = Column(Integer, primary_key=True)
    Username = Column(String(255), ForeignKey('User.Username'))
    Text = Column(Text, nullable=False)
    DatasetID = Column(Integer, ForeignKey('Dataset.DatasetID', ondelete="CASCADE"))  # Changed from ProjectID to DatasetID
    PositionInProject = Column(Integer)

    dataset = relationship("Dataset", back_populates="sentences")  # Changed from project to dataset
    segments = relationship("Segment", back_populates="sentence")

class Annotation(Base):
    __tablename__ = "Annotation"

    AnnotationID = Column(Integer, primary_key=True)
    AnnotationText = Column(Text, nullable=False)
    ParentAnnotationID = Column(Integer, ForeignKey('Annotation.AnnotationID', ondelete="CASCADE"))
    ProjectID = Column(Integer, ForeignKey('Project.ProjectID', ondelete="CASCADE"))

    project = relationship("Project", back_populates="annotations")
    segments = relationship("Segment", back_populates="annotation")


class Segment(Base):
    __tablename__ = "Segment"

    SegmentID = Column(Integer, primary_key=True)
    SentenceID = Column(Integer, ForeignKey('Sentence.SentenceID', ondelete="CASCADE"))
    Text = Column(Text, nullable=False)
    StartPosition = Column(Integer, nullable=False)
    AnnotationID = Column(Integer, ForeignKey('Annotation.AnnotationID', ondelete="CASCADE"))

    sentence = relationship("Sentence", back_populates="segments")
    embedding = relationship("Embedding", back_populates="segment")
    annotation = relationship("Annotation", back_populates="segments")


class Embedding(Base):
    __tablename__ = "Embedding"

    EmbeddingID = Column(Integer, primary_key=True)
    SegmentID = Column(Integer, ForeignKey('Segment.SegmentID', ondelete="CASCADE"))
    ModelID = Column(Integer, ForeignKey('CombinedModel.ModelID'))  # Changed from ModelName to ModelID
    EmbeddingValues = Column(LargeBinary, nullable=False)

    segment = relationship("Segment", back_populates="embedding")
    positions = relationship("Position", back_populates="embedding")
    combined_model = relationship("CombinedModel") # New relationship


class Position(Base):
    __tablename__ = "Position"

    PositionID = Column(Integer, primary_key=True)
    EmbeddingID = Column(Integer, ForeignKey('Embedding.EmbeddingID', ondelete="CASCADE"))
    ModelID = Column(Integer, ForeignKey('CombinedModel.ModelID'))
    Posx = Column(Float, nullable=False)
    Posy = Column(Float, nullable=False)

    embedding = relationship("Embedding", back_populates="positions")
    combined_model = relationship("CombinedModel")


class CombinedModel(Base):
    __tablename__ = "CombinedModel"

    ModelID = Column(Integer, primary_key=True)
    ProjectID = Column(Integer, ForeignKey('Project.ProjectID', ondelete="CASCADE"))
    ModelName = Column(String(255), nullable=False)
    ModelDescription = Column(Text)
    ModelFile = Column(String(1000), nullable=False)

    project = relationship("Project", back_populates="combined_models")

    __table_args__ = (UniqueConstraint('ProjectID', 'ModelName', name='_project_modelname_uc'),)


class View(Base):
    __tablename__ = "View"

    ViewID = Column(Integer, primary_key=True)
    ProjectID = Column(Integer, ForeignKey('Project.ProjectID', ondelete="CASCADE"))
    EmbeddingModelID = Column(Integer, ForeignKey('CombinedModel.ModelID'))  # Changed from ModelName to ModelID
    ReductionModelID = Column(Integer, ForeignKey('CombinedModel.ModelID'))  # Changed from ModelName to ModelID


    project = relationship("Project", back_populates="views")
    embedding_combined_model = relationship("CombinedModel", foreign_keys=[EmbeddingModelID])
    reduction_combined_model = relationship("CombinedModel", foreign_keys=[ReductionModelID])