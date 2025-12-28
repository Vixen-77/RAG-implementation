import mongoose, { Schema } from "mongoose";
import Joi from "joi"

const UserSchema = new mongoose.Schema({
  username:{
    type:String,
    required:true,
    trim:true,
  },
  email:{
    type:String,
    required:true,
    trim:true,
    unique:true,
  },
  password:{
    type:String,
    required:true,
    trim:true,
    minlength:6,
  }
})

const registerValidation = (obj)=>{
    const schema = Joi.object({
       username:Joi.string().required().trim(),
       email:Joi.string().required().trim().email(),
       password:Joi.string().required().trim().min(6),
    })
    return schema.validate(obj)
}

const loginValidation = (obj)=>{
    const schema = Joi.object({
       email:Joi.string().required().trim().email(),
       password:Joi.string().required().trim().min(6),
    })
    return schema.validate(obj)
}

const User = mongoose.model("User" , UserSchema)

export {User ,registerValidation , loginValidation}