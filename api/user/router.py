from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from db.session import get_db
from api import auth_utilities
from api.user.schemas import User, Login
from db.models import User as UserModel

router = APIRouter()


@router.post("/sign_up", response_model=auth_utilities.Token)
def sign_up(user: User, db: Session = Depends(get_db)):
    hashed_password = auth_utilities.get_password_hash(user.password)

    new_user = UserModel(Username=user.username, Email=user.email, Password=hashed_password)
    db.add(new_user)
    db.commit()

    access_token = auth_utilities.create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/login", response_model=auth_utilities.Token)
def login(form_data: auth_utilities.OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
#def login(form_data: Login, db: Session = Depends(get_db)):
#def login(form_data: auth_utilities.OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):

    user = db.query(UserModel).filter(UserModel.Username == form_data.username).first()

    if not user or not auth_utilities.verify_password(form_data.password, user.Password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")

    access_token = auth_utilities.create_access_token(data={"sub": user.Username})
    return {"access_token": access_token, "token_type": "bearer"}
