import hashlib
import os
from typing import Literal

import numpy as np
from sqlalchemy import (
    Column, Integer, String, create_engine, Double
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.types import TypeDecorator, LargeBinary


class BitStringBinary(TypeDecorator):
    impl = LargeBinary(16)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None

        if isinstance(value, bytes):
            v = value

        if isinstance(value, np.ndarray):
            v = np.packbits(value).tobytes()

        if isinstance(value, str):
            val_int = int(value, 2)
            v = val_int.to_bytes(256, byteorder='big')

        return hashlib.md5(v).digest()

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return value.hex()
        # val_int = int.from_bytes(value, byteorder='big')
        # return bin(val_int)[2:].zfill(2048)


Base = declarative_base()


class AtomType(Base):
    __tablename__ = "atom_type"
    id = Column(Integer, autoincrement=True, primary_key=True)
    opls_num = Column(String(10), nullable=False)
    element = Column(String(10), nullable=False)
    mass = Column(Double)
    charge = Column(Double)
    sigma = Column(Double)
    epsilon = Column(Double)
    ptype = Column(String(10))
    hash_str = Column(BitStringBinary, unique=True, index=True, nullable=False)
    bond_type = Column(String(10))


class BondType(Base):
    __tablename__ = "bond_type"
    id = Column(Integer, autoincrement=True, primary_key=True)
    opls_i = Column(String(10), nullable=False)
    opls_j = Column(String(10), nullable=False)
    k = Column(Double)
    r0 = Column(Double)
    hash_str = Column(BitStringBinary, unique=True, index=True, nullable=False)
    ftype = Column(Integer)
    # __table_args__ = (UniqueConstraint('opls_i', 'opls_j', name='u_bond'),)


class AngleType(Base):
    __tablename__ = "angle_type"
    id = Column(Integer, autoincrement=True, primary_key=True)
    opls_i = Column(String(10), nullable=False)
    opls_j = Column(String(10), nullable=False)
    opls_k = Column(String(10), nullable=False)
    k = Column(Double)
    t0 = Column(Double)
    hash_str = Column(BitStringBinary, unique=True, index=True, nullable=False)
    ftype = Column(Integer)
    # __table_args__ = (UniqueConstraint('opls_i', 'opls_j', 'opls_k', name='u_angle'),)


class DihedralType(Base):
    __tablename__ = "dihedral_type"
    id = Column(Integer, autoincrement=True, primary_key=True)
    opls_i = Column(String(10), nullable=False)
    opls_j = Column(String(10), nullable=False)
    opls_k = Column(String(10), nullable=False)
    opls_l = Column(String(10), nullable=False)
    C0 = Column(Double)
    C1 = Column(Double)
    C2 = Column(Double)
    C3 = Column(Double)
    C4 = Column(Double)
    C5 = Column(Double)
    ftype = Column(Integer)
    hash_str = Column(BitStringBinary, unique=True, index=True, nullable=False)
    # __table_args__ = (
    #     UniqueConstraint('opls_i', 'opls_j', 'opls_k', 'opls_l',
    #                      name='u_dihedral'),
    # )


class ImproperType(Base):
    __tablename__ = "improper_type"
    id = Column(Integer, autoincrement=True, primary_key=True)
    opls_i = Column(String(10), nullable=False)
    opls_j = Column(String(10), nullable=False)
    opls_k = Column(String(10), nullable=False)
    opls_l = Column(String(10), nullable=False)
    k = Column(Double)
    psi0 = Column(Double)
    ftype = Column(Integer)
    hash_str = Column(BitStringBinary, unique=True, index=True, nullable=False)
    # __table_args__ = (
    #     UniqueConstraint('opls_i', 'opls_j', 'opls_k', 'opls_l',
    #                      name='u_improper'),
    # )


class OplsDB:
    def __init__(self, db_path: str = "opls.db", overwrite: bool = False):
        if overwrite and os.path.isfile(db_path):
            os.remove(db_path)
        uri = f"sqlite:///{db_path}"
        self.engine = create_engine(uri, echo=False, future=True)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine, future=True)
        self.session = Session()

    def _insert(self, obj: Base):
        from sqlalchemy.exc import IntegrityError
        self.session.add(obj)
        try:
            self.session.commit()
        except IntegrityError:
            self.session.rollback()

    def insert(self, data: dict[int: Base]):
        for obj in data.values():
            self.session.add(obj)
        self.session.commit()

    def search(self, target: Literal['atom', 'bond', 'angle', 'diheral', 'improper'], **kw) -> list[Base]:
        __models = {
            'atom': AtomType,
            'bond': BondType,
            'angle': AngleType,
            'dihedral': DihedralType,
            'improper': ImproperType
        }
        model: Base = __models[target]
        q = self.session.query(model)
        for k, v in kw.items():
            q = q.filter(getattr(model, k) == v)
        return q.all()

    def close(self):
        self.session.close()
